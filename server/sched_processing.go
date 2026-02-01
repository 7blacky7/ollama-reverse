// Package server - Scheduler Request-Verarbeitung
//
// Diese Datei enthaelt:
// - GetRunner: Runner fuer Request holen
// - Run: Scheduler starten
// - processPending: Ausstehende Requests verarbeiten
// - processCompleted: Abgeschlossene Requests verarbeiten
package server

import (
	"context"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

// GetRunner holt einen Runner fuer das angegebene Model.
// Context muss cancelled werden um refCount zu dekrementieren und Runner freizugeben
func (s *Scheduler) GetRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration) (chan *runnerRef, chan error) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// Multimodale Models brauchen mindestens 2048 Context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
	}

	s.loadedMu.Lock()
	runner := s.loaded[req.model.ModelPath]
	s.loadedMu.Unlock()
	if runner != nil && !runner.needsReload(c, req) {
		req.useLoadedRunner(runner, s.finishedReqCh)
	} else {
		select {
		case s.pendingReqCh <- req:
		default:
			req.errCh <- ErrMaxQueue
		}
	}
	return req.successCh, req.errCh
}

// Run startet die Scheduler-Goroutinen. Beendet wenn ctx.Done().
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")
	go func() {
		s.processPending(ctx)
	}()

	go func() {
		s.processCompleted(ctx)
	}()
}

// processPending verarbeitet ausstehende Requests in einer Schleife
func (s *Scheduler) processPending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()

	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// Andere Requests blockieren bis dieser laeuft
			pending.schedAttempts++

			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

			for {
				var runnerToExpire *runnerRef
				s.loadedMu.Lock()
				runner := s.loaded[pending.model.ModelPath]
				loadedCount := len(s.loaded)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				s.loadedMu.Unlock()

				if runner != nil {
					if runner.needsReload(ctx, pending) {
						slog.Debug("reloading", "runner", runner)
						runnerToExpire = runner
					} else {
						// Runner ist nutzbar, zurueckgeben
						logutil.Trace("using existing loaded runner", "model", pending.model.ModelPath)
						pending.useLoadedRunner(runner, s.finishedReqCh)
						break
					}
				} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Keine Models geladen oder unter MaxRunners
					// Aktuelle GPU-Liste holen
					var gpus []ml.DeviceInfo
					if pending.opts.NumGPU == 0 {
						gpus = []ml.DeviceInfo{}
					} else {
						logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
						gpus = s.getGpuFn(ctx, runnersSnapshot)
					}
					logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
					systemInfo := s.getSystemInfoFn()
					if maxRunners <= 0 {
						// Kein MaxRunners vom User, automatisch bestimmen
						if pending.opts.NumGPU == 0 {
							// Tatsaechliche GPU-Liste fuer Default-Max-Models holen
							logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
							g := s.getGpuFn(ctx, runnersSnapshot)
							maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
						} else {
							maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
						}
						slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "gpu_count", len(gpus))
					}

					// Image-Generation-Model pruefen vor GGML-Laden
					if slices.Contains(pending.model.Config.Capabilities, "image") {
						if s.loadImageGen(pending) {
							break
						}
						continue
					}

					// Model fuer Fitting laden
					logutil.Trace("loading model metadata", "model", pending.model.ModelPath)
					ggml, err := llm.LoadModel(pending.model.ModelPath, 1024)
					if err != nil {
						pending.errCh <- err
						break
					}

					// Freien Speicher basierend auf geladenen Models aktualisieren
					logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
					s.updateFreeSpace(gpus)

					if loadedCount == 0 {
						// Keine Models geladen. Model laden, bevorzugt Best-Fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						s.loadFn(pending, ggml, systemInfo, gpus, false)
						break
					}

					// Mehr als ein Model geladen, pruefen ob neues passt
					logutil.Trace("loading additional model", "model", pending.model.ModelPath)
					needEvict := s.loadFn(pending, ggml, systemInfo, gpus, true)
					if !needEvict {
						slog.Debug("new model fits with existing models, loading")
						break
					}

					runnerToExpire = s.findRunnerToUnload()
				}

				if runnerToExpire == nil {
					// Waehrend Ladeberechnung wurden Runner parallel entladen
					// Also nochmal versuchen, loadedCount sollte 0 sein
					slog.Debug("runner to expire was nil, retrying")
					continue
				}
				// Expiration triggern um nach Fertigstellung zu entladen
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "runner", runnerToExpire, "refCount", runnerToExpire.refCount)
				if runnerToExpire.expireTimer != nil {
					runnerToExpire.expireTimer.Stop()
					runnerToExpire.expireTimer = nil
				}
				runnerToExpire.sessionDuration = 0
				if runnerToExpire.refCount <= 0 {
					s.expiredCh <- runnerToExpire
				}
				runnerToExpire.refMu.Unlock()
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
			// Entlade-Event ohne ausstehende Requests ignorieren
			slog.Debug("ignoring unload event with no pending requests")
		}
	}
}

// processCompleted verarbeitet abgeschlossene Requests und expired Timer
func (s *Scheduler) processCompleted(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler completed loop")
			return
		case finished := <-s.finishedReqCh:
			s.loadedMu.Lock()
			runner := s.loaded[finished.model.ModelPath]
			s.loadedMu.Unlock()
			if runner == nil {
				slog.Error("finished request signal received after model unloaded", "modelPath", finished.model.ModelPath)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			if runner.refCount <= 0 {
				if runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
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
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				}
			}
			slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
			runner.refMu.Unlock()
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "runner", runner)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				slog.Debug("expired event with positive ref count, retrying", "runner", runner, "refCount", runner.refCount)
				go func(runner *runnerRef) {
					// Kann noch nicht entladen, aber Event nochmal einreihen
					time.Sleep(10 * time.Millisecond)
					s.expiredCh <- runner
				}(runner)
				runner.refMu.Unlock()
				continue
			}

			s.loadedMu.Lock()
			slog.Debug("got lock to unload expired event", "runner", runner)
			runnerToUnload := s.loaded[runner.modelPath]
			if runnerToUnload == nil {
				// runnerToUnload ist nil, wurde bereits entladen
				// Kann passieren bei abgebrochenem Request oder Reload
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
				slog.Debug("duplicate expired event, ignoring", "runner", runner)
			} else if runner.pid != runnerToUnload.pid {
				// PIDs stimmen nicht ueberein, wahrscheinlich mehrere Ladefehler
				// Verwaisten Runner herunterfahren, aber nicht aus loaded loeschen
				slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
				runner.unload()
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
			} else {
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
	}
}
