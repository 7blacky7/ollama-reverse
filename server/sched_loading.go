// Package server - Scheduler Model-Laden
//
// Diese Datei enthaelt:
// - load: Model laden
// - loadImageGen: Image-Generation-Model laden
package server

import (
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen"
)

// load erstellt und laedt ein neues Model. Wenn requireFull=true muss Model komplett auf GPUs passen.
// Gibt zurueck ob ein Model entladen werden muss damit dieses passt.
func (s *Scheduler) load(req *LlmRequest, f *ggml.GGML, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	numParallel := max(int(envconfig.NumParallel()), 1)

	// Embedding-Models immer mit parallel=1 laden
	if req.model.CheckCapabilities(model.CapabilityCompletion) != nil {
		numParallel = 1
	}

	// mllama, qwen3vl, qwen3vlmoe nutzen Encoder-Cache der nicht mit num_parallel > 1 funktioniert
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading

	if llama == nil {
		var err error
		llama, err = s.newServerFn(systemInfo, gpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts, numParallel)
		if err != nil {
			// Einige aeltere Models sind inkompatibel mit neueren llama.cpp Versionen
			if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
				err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
			}
			slog.Info("NewLlamaServer failed", "model", req.model.ModelPath, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		s.activeLoading = llama
	} else {
		if s.activeLoading.ModelPath() != req.model.ModelPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), req.model.ModelPath))
		}
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range gpus {
		available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()
		if gpu.FreeMemory < envconfig.GpuOverhead()+gpu.MinimumMemory() {
			available = 0
		}
		slog.Info("gpu memory", "id", gpu.ID, "library", gpu.Library,
			"available", format.HumanBytes2(available),
			"free", format.HumanBytes2(gpu.FreeMemory),
			"minimum", format.HumanBytes2(gpu.MinimumMemory()),
			"overhead", format.HumanBytes2(envconfig.GpuOverhead()))
	}

	gpuIDs, err := llama.Load(req.ctx, systemInfo, gpus, requireFull)
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// Keine anderen Models geladen, passt trotzdem nicht
				slog.Info("model is too large for system memory", "requireFull", requireFull)
				s.activeLoading.Close()
				s.activeLoading = nil
				req.errCh <- err
			}
			return true
		}

		slog.Info("Load failed", "model", req.model.ModelPath, "error", err)
		s.activeLoading.Close()
		s.activeLoading = nil
		req.errCh <- err
		return false
	}

	// Pruefen ob diskrete GPUs vorhanden fuer VRAM-Monitoring beim Shutdown
	discreteGPUs := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range gpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					discreteGPUs = true
					break iGPUScan
				}
			}
		}
	}

	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		discreteGPUs:    discreteGPUs,
		vramSize:        llama.VRAMSize(),
		totalSize:       llama.TotalSize(),
		loading:         true,
		pid:             llama.Pid(),
	}
	runner.numParallel = numParallel
	runner.refMu.Lock() // Lock halten bis running oder abgebrochen

	s.loadedMu.Lock()
	if oldRunner, ok := s.loaded[req.model.ModelPath]; ok {
		// Sollte nicht passieren, aber Absicherung gegen Runner-Leak
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
		oldRunner.refMu.Lock()
		oldRunner.unload()
		oldRunner.refMu.Unlock()
	}
	s.activeLoading = nil
	s.loaded[req.model.ModelPath] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()

	go func() {
		defer runner.refMu.Unlock()
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			slog.Error("error loading llama server", "error", err)
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "runner", runner)
			s.expiredCh <- runner
			return
		}
		slog.Debug("finished setting up", "runner", runner)
		if runner.pid < 0 {
			runner.pid = llama.Pid()
		}
		runner.refCount++
		runner.loading = false
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
	}()

	return false
}

// loadImageGen laedt ein Image-Generation-Model
func (s *Scheduler) loadImageGen(req *LlmRequest) bool {
	// Model-Name fuer imagegen verwenden (loest Manifests nach Name auf)
	modelName := req.model.ShortName
	server, err := imagegen.NewServer(modelName)
	if err != nil {
		req.errCh <- err
		return true
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		llama:           server,
		Options:         &req.opts,
		loading:         false,
		sessionDuration: sessionDuration,
		totalSize:       server.TotalSize(),
		vramSize:        server.VRAMSize(),
	}

	s.loadedMu.Lock()
	s.loaded[req.model.ModelPath] = runner
	s.loadedMu.Unlock()

	// Expiration-Timer setzen
	runner.refMu.Lock()
	if sessionDuration > 0 {
		runner.expireTimer = time.AfterFunc(sessionDuration, func() {
			s.expiredCh <- runner
		})
	}
	runner.refMu.Unlock()

	req.useLoadedRunner(runner, s.finishedReqCh)
	return true
}
