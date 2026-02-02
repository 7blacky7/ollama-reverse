// Package server - Scheduler Request-Verarbeitung
//
// Diese Datei enthaelt:
// - processPending: Ausstehende Requests verarbeiten
// - processCompleted: Abgeschlossene Requests und Timer verarbeiten
//
// processPending ist die Hauptschleife fuer neue Requests.
// Sie entscheidet ob ein Runner wiederverwendet, neu geladen
// oder ein bestehender entladen werden muss.
package server

import (
	"context"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// processPending verarbeitet ausstehende Requests in einer Endlosschleife.
// Fuer jeden Request wird entschieden:
// 1. Bestehender Runner kann wiederverwendet werden
// 2. Neuer Runner muss geladen werden
// 3. Bestehender Runner muss entladen werden um Platz zu schaffen
func (s *Scheduler) processPending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()

	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return

		case pending := <-s.pendingReqCh:
			// Scheduling-Versuch zaehlen
			pending.schedAttempts++

			// Abgebrochene Requests ueberspringen
			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

			// Request-Verarbeitungsschleife
			for {
				var runnerToExpire *runnerRef

				// Snapshot der geladenen Runner erstellen
				s.loadedMu.Lock()
				runner := s.loaded[pending.model.ModelPath]
				loadedCount := len(s.loaded)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				s.loadedMu.Unlock()

				if runner != nil {
					// Runner fuer dieses Model existiert bereits
					if runner.needsReload(ctx, pending) {
						slog.Debug("reloading", "runner", runner)
						runnerToExpire = runner
					} else {
						// Runner direkt nutzen
						logutil.Trace("using existing loaded runner", "model", pending.model.ModelPath)
						pending.useLoadedRunner(runner, s.finishedReqCh)
						break
					}
				} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
					// MaxRunners erreicht - Platz schaffen
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Neuen Runner laden
					runnerToExpire = s.tryLoadNewRunner(ctx, pending, &maxRunners, loadedCount, runnersSnapshot)
					if runnerToExpire == nil && pending.ctx.Err() == nil {
						// Model wurde geladen oder Fehler aufgetreten
						break
					}
				}

				// Sicherheitspruefung
				if runnerToExpire == nil {
					slog.Debug("runner to expire was nil, retrying")
					continue
				}

				// Runner zum Entladen markieren
				s.expireRunner(runnerToExpire)

				// Auf Entladen warten
				slog.Debug("waiting for pending requests to complete and unload to occur", "runner", runnerToExpire)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "runner", runnerToExpire)
					continue
				}
			}

		case <-s.unloadedCh:
			// Entlade-Event ignorieren wenn keine Requests anstehen
			slog.Debug("ignoring unload event with no pending requests")
		}
	}
}

// tryLoadNewRunner versucht einen neuen Runner zu laden.
// Gibt einen Runner zum Entladen zurueck wenn kein Platz ist, sonst nil.
func (s *Scheduler) tryLoadNewRunner(ctx context.Context, pending *LlmRequest, maxRunners *uint, loadedCount int, runnersSnapshot []ml.FilteredRunnerDiscovery) *runnerRef {
	// GPU-Liste holen
	var gpus []ml.DeviceInfo
	if pending.opts.NumGPU == 0 {
		gpus = []ml.DeviceInfo{}
	} else {
		logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
		gpus = s.getGpuFn(ctx, runnersSnapshot)
	}

	logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
	systemInfo := s.getSystemInfoFn()

	// MaxRunners automatisch bestimmen wenn nicht gesetzt
	if *maxRunners <= 0 {
		if pending.opts.NumGPU == 0 {
			logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
			g := s.getGpuFn(ctx, runnersSnapshot)
			*maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
		} else {
			*maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
		}
		slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", *maxRunners, "gpu_count", len(gpus))
	}

	// Image-Generation Model Sonderbehandlung
	if slices.Contains(pending.model.Config.Capabilities, "image") {
		if s.loadImageGen(pending) {
			return nil
		}
		// ImageGen Laden fehlgeschlagen - nochmal versuchen
		return nil
	}

	// GGML Model laden
	logutil.Trace("loading model metadata", "model", pending.model.ModelPath)
	ggml, err := llm.LoadModel(pending.model.ModelPath, 1024)
	if err != nil {
		pending.errCh <- err
		return nil
	}

	// Freien GPU-Speicher aktualisieren
	logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
	s.updateFreeSpace(gpus)

	if loadedCount == 0 {
		// Erstes Model - direkt laden
		slog.Debug("loading first model", "model", pending.model.ModelPath)
		s.loadFn(pending, ggml, systemInfo, gpus, false)
		return nil
	}

	// Pruefen ob neues Model neben bestehenden passt
	logutil.Trace("loading additional model", "model", pending.model.ModelPath)
	needEvict := s.loadFn(pending, ggml, systemInfo, gpus, true)
	if !needEvict {
		slog.Debug("new model fits with existing models, loading")
		return nil
	}

	// Kein Platz - Runner zum Entladen finden
	return s.findRunnerToUnload()
}

// expireRunner markiert einen Runner zum sofortigen Entladen
func (s *Scheduler) expireRunner(runner *runnerRef) {
	runner.refMu.Lock()
	slog.Debug("resetting model to expire immediately to make room", "runner", runner, "refCount", runner.refCount)

	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0

	if runner.refCount <= 0 {
		s.expiredCh <- runner
	}
	runner.refMu.Unlock()
}

// processCompleted verarbeitet abgeschlossene Requests und expired Timer.
// Verwaltet Runner-Lifecycle: RefCount-Updates, Timer-Management, Entladen.
func (s *Scheduler) processCompleted(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler completed loop")
			return

		case finished := <-s.finishedReqCh:
			s.handleFinishedRequest(finished)

		case runner := <-s.expiredCh:
			s.handleExpiredRunner(runner)
		}
	}
}

// handleFinishedRequest verarbeitet ein abgeschlossenes Request-Event
func (s *Scheduler) handleFinishedRequest(finished *LlmRequest) {
	s.loadedMu.Lock()
	runner := s.loaded[finished.model.ModelPath]
	s.loadedMu.Unlock()

	if runner == nil {
		slog.Error("finished request signal received after model unloaded", "modelPath", finished.model.ModelPath)
		return
	}

	runner.refMu.Lock()
	runner.refCount--

	if runner.refCount <= 0 {
		s.handleIdleRunner(runner)
	}

	slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
	runner.refMu.Unlock()
}

// handleIdleRunner verarbeitet einen Runner der gerade idle geworden ist
func (s *Scheduler) handleIdleRunner(runner *runnerRef) {
	if runner.sessionDuration <= 0 {
		// Sofort entladen bei Null-Duration
		slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		s.expiredCh <- runner
	} else if runner.expireTimer == nil {
		// Neuen Timer starten
		slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
		runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
			slog.Debug("timer expired, expiring to unload", "runner", runner)
			runner.refMu.Lock()
			defer runner.refMu.Unlock()
			if runner.expireTimer != nil {
				runner.expireTimer.Stop()
				runner.expireTimer = nil
			}
			s.expiredCh <- runner
		})
		runner.expiresAt = time.Now().Add(runner.sessionDuration)
	} else {
		// Bestehenden Timer zuruecksetzen
		slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
		runner.expireTimer.Reset(runner.sessionDuration)
		runner.expiresAt = time.Now().Add(runner.sessionDuration)
	}
}

// handleExpiredRunner verarbeitet ein Runner-Expiration Event
func (s *Scheduler) handleExpiredRunner(runner *runnerRef) {
	slog.Debug("runner expired event received", "runner", runner)
	runner.refMu.Lock()

	if runner.refCount > 0 {
		// Runner noch in Verwendung - spaeter nochmal versuchen
		slog.Debug("expired event with positive ref count, retrying", "runner", runner, "refCount", runner.refCount)
		go func(runner *runnerRef) {
			time.Sleep(10 * time.Millisecond)
			s.expiredCh <- runner
		}(runner)
		runner.refMu.Unlock()
		return
	}

	// Runner entladen
	s.loadedMu.Lock()
	slog.Debug("got lock to unload expired event", "runner", runner)
	runnerToUnload := s.loaded[runner.modelPath]

	if runnerToUnload == nil {
		// Bereits entladen
		s.loadedMu.Unlock()
		runner.refMu.Unlock()
		slog.Debug("duplicate expired event, ignoring", "runner", runner)
	} else if runner.pid != runnerToUnload.pid {
		// PID-Mismatch - verwaisten Runner herunterfahren
		slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
		runner.unload()
		s.loadedMu.Unlock()
		runner.refMu.Unlock()
	} else {
		// Runner entladen und VRAM-Recovery abwarten
		slog.Debug("starting background wait for VRAM recovery", "runner", runner)
		runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
		for _, r := range s.loaded {
			runnersSnapshot = append(runnersSnapshot, r)
		}
		finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
		runner.unload()
		delete(s.loaded, runner.modelPath)
		s.loadedMu.Unlock()

		slog.Debug("runner terminated and removed from list, blocking for VRAM recovery", "runner", runner)
		<-finished
		runner.refMu.Unlock()

		slog.Debug("sending an unloaded event", "runner", runner)
		s.unloadedCh <- struct{}{}
	}
}
