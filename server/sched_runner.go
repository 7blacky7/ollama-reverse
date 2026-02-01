// Package server - Scheduler Runner-Hilfsfunktionen
//
// Diese Datei enthaelt:
// - useLoadedRunner: Runner an Request uebergeben
// - needsReload: Pruefen ob Runner neu geladen werden muss
// - findRunnerToUnload: Runner zum Entladen finden
package server

import (
	"context"
	"log/slog"
	"reflect"
	"sort"
	"time"
)

// useLoadedRunner uebergibt Runner an Request und richtet Finished-Event ein
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) {
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	runner.refCount++
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
}

// needsReload prueft ob ein Runner fuer den Request neu geladen werden muss
func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", req.model.ModelPath)
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	timeout := 10 * time.Second
	if runner.loading {
		timeout = 2 * time.Minute // Initiales Laden kann lange dauern
	}

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

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) || // Adapter geaendert?
		!reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) || // Projectors geaendert?
		!reflect.DeepEqual(optsExisting, optsNew) || // Runner-Optionen geaendert?
		runner.llama.Ping(ctx) != nil {
		return true
	}

	return false
}

// findRunnerToUnload findet einen Runner zum Entladen
func (s *Scheduler) findRunnerToUnload() *runnerRef {
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

	// Zukuenftig koennte der Algorithmus optimiert werden um besseren Runner zu waehlen
	sort.Sort(ByDurationAndName(runnerList))

	// Zuerst idle Runner suchen
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload", "runner", runner)
			return runner
		}
	}
	// Keiner idle, den mit kuerzester Dauer nehmen
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
	return runnerList[0]
}
