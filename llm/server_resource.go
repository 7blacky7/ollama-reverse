// Package llm - Resource Management
//
// Funktionen zur Verwaltung von Server-Ressourcen:
// - Close: Server herunterfahren und Speicher freigeben
// - GetDeviceInfos: GPU-Informationen abrufen
// - VRAMSize/TotalSize/VRAMByGPU: Speichernutzung abfragen
package llm

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/ml"
)

// Close beendet den Server und gibt alle Ressourcen frei
func (s *llmServer) Close() error {
	s.llamaModelLock.Lock()
	if s.llamaModel != nil {
		llama.FreeModel(s.llamaModel)
		s.llamaModel = nil
	}
	s.llamaModelLock.Unlock()

	if s.cmd != nil {
		slog.Debug("stopping llama server", "pid", s.Pid())
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}
		// Wenn ProcessState bereits gesetzt, ist Wait abgeschlossen
		if s.cmd.ProcessState == nil {
			slog.Debug("waiting for llama server to exit", "pid", s.Pid())
			<-s.done
		}

		slog.Debug("llama server stopped", "pid", s.Pid())
	}

	return nil
}

// GetDeviceInfos gibt GPU-Informationen zurück (llamaServer)
// Nicht unterstützt für llama runner
func (s *llamaServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	slog.Debug("llamarunner free vram reporting not supported")
	return nil
}

// GetDeviceInfos gibt GPU-Informationen zurück (ollamaServer)
func (s *ollamaServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	devices, err := ml.GetDevicesFromRunner(ctx, s)
	if err != nil {
		if s.cmd != nil && s.cmd.ProcessState == nil {
			slog.Debug("failure refreshing GPU information", "error", err)
		}
	}
	return devices
}

// VRAMSize gibt die gesamte VRAM-Nutzung über alle GPUs zurück
func (s *llmServer) VRAMSize() uint64 {
	if s.mem == nil {
		return 0
	}

	var mem uint64
	for _, g := range s.mem.GPUs {
		mem += g.Size()
	}

	// Bei vollständigem GPU-Offloading auch CPU-Komponenten zählen
	noCPULayers := true
	for i := range s.mem.CPU.Weights {
		if s.mem.CPU.Weights[i] != 0 || s.mem.CPU.Cache[i] != 0 {
			noCPULayers = false
			break
		}
	}
	if noCPULayers {
		mem += s.mem.InputWeights
		mem += s.mem.CPU.Graph
	}

	return mem
}

// TotalSize gibt die gesamte Speichernutzung zurück (CPU + GPU)
func (s *llmServer) TotalSize() uint64 {
	if s.mem == nil {
		return 0
	}

	mem := s.mem.InputWeights
	mem += s.mem.CPU.Size()
	for _, g := range s.mem.GPUs {
		mem += g.Size()
	}

	return mem
}

// VRAMByGPU gibt die VRAM-Nutzung einer spezifischen GPU zurück
func (s *llmServer) VRAMByGPU(id ml.DeviceID) uint64 {
	if s.mem == nil {
		return 0
	}

	for _, g := range s.mem.GPUs {
		if g.DeviceID == id {
			return g.Size()
		}
	}

	return 0
}
