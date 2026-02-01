// Package server - Scheduler Speicherverwaltung
//
// Diese Datei enthaelt:
// - updateFreeSpace: Freien GPU-Speicher aktualisieren
// - waitForVRAMRecovery: Auf VRAM-Freigabe warten
// - unloadAllRunners: Alle Runner entladen
// - expireRunner: Runner zum Ablauf markieren
package server

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// updateFreeSpace aktualisiert den freien Speicher basierend auf geladenen Models
func (s *Scheduler) updateFreeSpace(allGpus []ml.DeviceInfo) {
	if len(allGpus) == 0 {
		return
	}
	predMap := map[ml.DeviceID]uint64{} // Summierte Nutzung pro GPU
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		r.refMu.Lock()
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[gpu.DeviceID] += r.llama.VRAMByGPU(gpu.DeviceID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
	}

	// GPU-Liste mit summierten Nutzungsprognosen aktualisieren
	for i := range allGpus {
		if p, ok := predMap[allGpus[i].DeviceID]; ok {
			slog.Debug("gpu reported", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "available", format.HumanBytes2(allGpus[i].FreeMemory))
			if p > allGpus[i].TotalMemory {
				// Sollte nicht passieren
				slog.Warn("predicted usage exceeds VRAM", "gpu", allGpus[i].ID, "totalMemory", allGpus[i].TotalMemory, "predicted", p)
				allGpus[i].FreeMemory = 0
			} else if (allGpus[i].TotalMemory - p) < allGpus[i].FreeMemory {
				// Prognostiziert frei ist kleiner als gemeldet frei, nutzen
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM based on existing loaded models", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

// waitForVRAMRecovery wartet bis VRAM nach Runner-Ende wieder frei ist.
// Muss vor unload aufgerufen werden um Baseline zu etablieren.
// Gibt Channel zurueck der benachrichtigt wenn fertig oder Timeout.
func (s *Scheduler) waitForVRAMRecovery(runner *runnerRef, runners []ml.FilteredRunnerDiscovery) chan any {
	finished := make(chan any, 1)

	// CPU, Metal und iGPUs brauchen keine Pruefung
	if len(runner.gpus) == 0 || !runner.discreteGPUs ||
		(len(runner.gpus) == 1 && runner.gpus[0].Library == "Metal") {
		finished <- struct{}{}
		slog.Debug("no need to wait for VRAM recovery", "runner", runner)
		return finished
	}
	start := time.Now()

	// Baseline vor Entladen etablieren
	gpusBefore := s.getGpuFn(context.Background(), runners)
	var totalMemoryBefore, freeMemoryBefore uint64
	for _, gpu := range gpusBefore {
		totalMemoryBefore += gpu.TotalMemory
		freeMemoryBefore += gpu.FreeMemory
	}
	totalMemoryNow := totalMemoryBefore
	freeMemoryNow := freeMemoryBefore

	go func() {
		// Typische Konvergenz 0.5-1.5s - Bei zu langer Dauer Scheduler VRAM schaetzen lassen
		ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
		defer cancel()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// GPUs abfragen, auf Free-Anstieg pruefen
				gpusNow := s.getGpuFn(ctx, runners)
				totalMemoryNow = 0
				freeMemoryNow = 0
				for _, gpu := range gpusNow {
					totalMemoryNow += gpu.TotalMemory
					freeMemoryNow += gpu.FreeMemory
				}
				if freeMemoryNow > freeMemoryBefore {
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(runner.vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// Bei ~75% der geschaetzten Nutzung wiederhergestellt, beenden
				if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
					slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
					finished <- struct{}{}
					return
				}
			case <-ctx.Done():
				slog.Debug("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

// unloadAllRunners entlaedt alle Runner
func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	if s.activeLoading != nil {
		slog.Debug("shutting down currently loading runner")
		s.activeLoading.Close()
		s.activeLoading = nil
	}

	for model, runner := range s.loaded {
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", model)
			runner.llama.Close()
		}
	}
}

// expireRunner markiert einen Runner zum sofortigen Ablauf
func (s *Scheduler) expireRunner(model *Model) {
	s.loadedMu.Lock()
	runner, ok := s.loaded[model.ModelPath]
	s.loadedMu.Unlock()
	if ok {
		runner.refMu.Lock()
		runner.expiresAt = time.Now()
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
}
