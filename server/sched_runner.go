// Package server - Scheduler Runner-Verwaltung
//
// Diese Datei enthaelt:
// - GetRunner: Runner fuer ein Model anfordern
// - Run: Scheduler-Goroutinen starten
// - useLoadedRunner: Runner an Request uebergeben
// - needsReload: Pruefen ob Runner neu geladen werden muss
// - findRunnerToUnload: Runner zum Entladen finden
//
// GetRunner ist der Haupteinstiegspunkt fuer Request-Verarbeitung.
// Es prueft ob ein passender Runner geladen ist und gibt diesen zurueck
// oder reiht den Request in die Pending-Queue ein.
package server

import (
	"context"
	"log/slog"
	"reflect"
	"sort"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// GetRunner holt einen Runner fuer das angegebene Model.
// Der zurueckgegebene Context muss gecancelled werden um den refCount
// zu dekrementieren und den Runner fuer andere Requests freizugeben.
//
// Rueckgabe:
// - successCh: Kanal auf dem der Runner zurueckgegeben wird
// - errCh: Kanal fuer Fehler (z.B. ErrMaxQueue wenn Queue voll)
func (s *Scheduler) GetRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration) (chan *runnerRef, chan error) {
	// Minimaler Context von 4 Tokens
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	// Multimodale Models benoetigen mindestens 2048 Context
	// fuer die Verarbeitung von Bildern und anderen Medien
	if m.CheckCapabilities(model.CapabilityVision) == nil {
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	// Request-Objekt erstellen
	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
	}

	// Pruefen ob bereits ein passender Runner geladen ist
	s.loadedMu.Lock()
	runner := s.loaded[req.model.ModelPath]
	s.loadedMu.Unlock()

	if runner != nil && !runner.needsReload(c, req) {
		// Runner ist geladen und passt - direkt verwenden
		req.useLoadedRunner(runner, s.finishedReqCh)
	} else {
		// Kein passender Runner - Request in Queue einreihen
		select {
		case s.pendingReqCh <- req:
			// Request erfolgreich eingereiht
		default:
			// Queue ist voll
			req.errCh <- ErrMaxQueue
		}
	}

	return req.successCh, req.errCh
}

// Run startet die Scheduler-Goroutinen.
// Beendet automatisch wenn ctx.Done() signalisiert wird.
//
// Startet zwei Goroutinen:
// - processPending: Verarbeitet neue Requests aus der Queue
// - processCompleted: Verarbeitet abgeschlossene Requests und Timeouts
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")

	// Goroutine fuer ausstehende Requests
	go func() {
		s.processPending(ctx)
	}()

	// Goroutine fuer abgeschlossene Requests
	go func() {
		s.processCompleted(ctx)
	}()
}

// useLoadedRunner uebergibt den Runner an den Request und richtet das
// Finished-Event ein. Inkrementiert refCount und stoppt den Expire-Timer.
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) {
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	// RefCount erhoehen da Runner jetzt in Verwendung
	runner.refCount++

	// Expire-Timer stoppen solange Runner aktiv
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}

	// Session-Duration vom Request uebernehmen wenn angegeben
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}

	// Runner an Request uebergeben
	pending.successCh <- runner

	// Finished-Event einrichten fuer Context-Ende
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
}

// needsReload prueft ob ein Runner fuer den Request neu geladen werden muss.
// Gruende fuer Reload: Adapter geaendert, Projectors geaendert,
// Runner-Optionen geaendert, oder Runner antwortet nicht auf Ping.
func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", req.model.ModelPath)
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	// Timeout fuer Ping - laenger wenn Runner noch laedt
	timeout := 10 * time.Second
	if runner.loading {
		timeout = 2 * time.Minute // Initiales Laden kann lange dauern
	}

	// Keine Options gesetzt - Reload noetig
	if runner.Options == nil {
		return true
	}

	// Runner nicht neu laden wenn num_gpu=-1 angegeben wurde
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	// Konfigurationsvergleich und Health-Check
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) ||
		!reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) ||
		!reflect.DeepEqual(optsExisting, optsNew) ||
		runner.llama.Ping(ctx) != nil {
		return true
	}

	return false
}

// findRunnerToUnload findet einen Runner der entladen werden kann.
// Bevorzugt idle Runner (refCount=0), ansonsten den mit kuerzester Duration.
func (s *Scheduler) findRunnerToUnload() *runnerRef {
	// Snapshot der geladenen Runner erstellen
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()

	if len(runnerList) == 0 {
		slog.Debug("no loaded runner to unload")
		return nil
	}

	// Nach Duration und Name sortieren fuer deterministische Auswahl
	sort.Sort(ByDurationAndName(runnerList))

	// Zuerst nach idle Runnern suchen (refCount == 0)
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload", "runner", runner)
			return runner
		}
	}

	// Kein idle Runner gefunden - den mit kuerzester Duration nehmen
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
	return runnerList[0]
}
