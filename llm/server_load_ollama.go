// Package llm - Model Loading (ollamaServer)
//
// Iteratives Model-Loading für Ollama Engine:
// - ollamaServer.Load: Hauptfunktion
// - iterativeLoad: Durchläuft fit -> alloc -> commit
// - loadOperation: Führt einzelne Operation aus
// - findStableLayout: Sucht stabiles Memory-Layout
// - exploreIntermediateLayers: Testet Zwischenzustände
package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/ml"
)

// Load lädt das Model iterativ (ollamaServer Variante)
// Iteriert durch fit -> alloc -> commit Stufen
func (s *ollamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	var success bool
	defer func() {
		if !success {
			s.initModel(ctx, LoadRequest{}, LoadOperationClose)
		}
		if s.mem != nil {
			s.mem.Log(slog.LevelInfo)
		}
	}()

	slog.Info("loading model", "model layers", s.totalLayers, "requested", s.options.NumGPU)

	pastAllocations := make(map[uint64]struct{})
	var backoff float32

	gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
	if err != nil {
		return nil, err
	}

	if err := s.waitUntilRunnerLaunched(ctx); err != nil {
		return nil, err
	}

	gpuLayers, err = s.iterativeLoad(ctx, systemInfo, gpus, requireFull, gpuLayers, pastAllocations, backoff)
	if err != nil {
		return nil, err
	}

	// Final commit
	s.loadRequest.GPULayers = gpuLayers
	resp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)
	if err != nil {
		return nil, err
	}

	success = resp.Success
	s.mem = &resp.Memory

	if !success {
		slog.Warn("failed to commit memory for model", "memory", resp.Memory)
		return nil, errors.New("failed to commit memory for model")
	}

	return uniqueDeviceIDs(gpuLayers), nil
}

// iterativeLoad führt alle Load-Operationen der Reihe nach aus
func (s *ollamaServer) iterativeLoad(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool, gpuLayers ml.GPULayersList, pastAllocations map[uint64]struct{}, backoff float32) (ml.GPULayersList, error) {
	for operation := LoadOperationFit; operation < LoadOperationCommit; operation++ {
		var err error
		gpuLayers, err = s.loadOperation(ctx, systemInfo, gpus, requireFull, operation, gpuLayers, pastAllocations, &backoff)
		if err != nil {
			return nil, err
		}
	}
	return gpuLayers, nil
}

// loadOperation führt eine einzelne Load-Operation aus
func (s *ollamaServer) loadOperation(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool, operation LoadOperation, gpuLayers ml.GPULayersList, pastAllocations map[uint64]struct{}, backoff *float32) (ml.GPULayersList, error) {
	for {
		s.loadRequest.GPULayers = gpuLayers
		resp, err := s.initModel(ctx, s.loadRequest, operation)
		if err != nil {
			return nil, err
		}

		resp.Memory.Log(slog.LevelDebug)
		slog.Debug("memory", "success", resp.Success, "required", resp.Memory)

		pastAllocations[gpuLayers.Hash()] = struct{}{}
		s.mem = &resp.Memory

		newLayers, converged, err := s.findStableLayout(ctx, systemInfo, gpus, requireFull, operation, gpuLayers, pastAllocations, backoff, resp)
		if err != nil {
			return nil, err
		}

		if converged {
			return newLayers, nil
		}

		gpuLayers = newLayers
	}
}

// findStableLayout sucht ein stabiles Memory-Layout
func (s *ollamaServer) findStableLayout(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool, operation LoadOperation, gpuLayers ml.GPULayersList, pastAllocations map[uint64]struct{}, backoff *float32, resp *LoadResponse) (ml.GPULayersList, bool, error) {
	for {
		newGPULayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, *backoff)
		if err != nil {
			return nil, false, err
		}

		slog.Debug("new layout created", "layers", newGPULayers)

		// Neues Layout gefunden das noch nicht probiert wurde
		if _, ok := pastAllocations[newGPULayers.Hash()]; !ok && newGPULayers.Sum() <= gpuLayers.Sum() {
			return newGPULayers, false, nil
		}

		// Intermediate Layer Exploration bei großen Unterschieden
		if s.options.NumGPU < 0 && newGPULayers.Sum()-gpuLayers.Sum() > 1 {
			result, err := s.exploreIntermediateLayers(ctx, systemInfo, gpus, requireFull, operation, gpuLayers, newGPULayers, pastAllocations, *backoff)
			if err != nil {
				return nil, false, err
			}
			if result != nil {
				return result, true, nil
			}
		}

		if resp.Success {
			return gpuLayers, true, nil
		}

		if s.options.NumGPU >= 0 {
			return nil, false, fmt.Errorf("memory layout cannot be allocated with num_gpu = %v", s.options.NumGPU)
		}

		// Backoff erhöhen wenn Layout nicht passt
		if *backoff > 1 {
			slog.Warn("memory layout cannot be allocated", "memory", resp.Memory)
			return nil, false, errors.New("memory layout cannot be allocated")
		}

		*backoff += 0.1
		slog.Info("model layout did not fit, applying backoff", "backoff", fmt.Sprintf("%.2f", *backoff))
	}
}

// exploreIntermediateLayers testet Zwischenzustände bei Layoutänderungen
func (s *ollamaServer) exploreIntermediateLayers(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool, operation LoadOperation, gpuLayers, newGPULayers ml.GPULayersList, pastAllocations map[uint64]struct{}, backoff float32) (ml.GPULayersList, error) {
	for i := newGPULayers.Sum() - 1; i >= gpuLayers.Sum(); i-- {
		slog.Debug("exploring intermediate layers", "layer", i)

		s.options.NumGPU = i
		testLayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
		s.options.NumGPU = -1
		if err != nil {
			return nil, err
		}

		slog.Debug("new layout created", "layers", testLayers)

		s.loadRequest.GPULayers = testLayers
		resp, err := s.initModel(ctx, s.loadRequest, operation)
		if err != nil {
			return nil, err
		}

		resp.Memory.Log(slog.LevelDebug)
		slog.Debug("memory", "success", resp.Success, "required", resp.Memory)

		if resp.Success {
			verifyLayers, err := s.createLayout(systemInfo, gpus, &resp.Memory, requireFull, backoff)
			if err != nil {
				return nil, err
			}

			slog.Debug("verifying layout", "layers", verifyLayers)

			if testLayers.Sum() <= verifyLayers.Sum() {
				clear(pastAllocations)
				return testLayers, nil
			}
		}
	}

	return nil, nil
}
