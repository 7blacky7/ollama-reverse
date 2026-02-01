// runner_load.go - Model Loading HTTP Handler fuer den Ollama Runner
//
// Enthaelt:
// - load: HTTP Handler fuer Load-Operationen
// - info: HTTP Handler fuer Backend-Device Informationen

package ollamarunner

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// load ist der HTTP Handler fuer Load-Operationen (vom Ollama Server aufgerufen)
func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	if s.status != llm.ServerStatusLaunched {
		http.Error(w, "model already loaded", http.StatusInternalServerError)
		return
	}

	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	slog.Info("load", "request", req)

	if req.Operation == llm.LoadOperationClose {
		s.closeModel()
		if err := json.NewEncoder(w).Encode(&llm.LoadResponse{}); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.lastLoad.Operation = req.Operation
	loadModel := s.model == nil || !reflect.DeepEqual(req, s.lastLoad)

	s.lastLoad = req

	if loadModel {
		s.closeModel()

		params := ml.BackendParams{
			AllocMemory:    req.Operation != llm.LoadOperationFit,
			NumThreads:     req.NumThreads,
			GPULayers:      req.GPULayers,
			FlashAttention: req.FlashAttention,
		}

		s.batchSize = req.BatchSize

		err := s.allocModel(s.modelPath, params, req.LoraPath, req.Parallel, req.KvCacheType, req.KvSize, req.MultiUserCache)
		if err != nil {
			s.closeModel()

			var noMem ml.ErrNoMem
			if errors.As(err, &noMem) {
				resp := llm.LoadResponse{Success: false, Memory: noMem.BackendMemory}
				if err := json.NewEncoder(w).Encode(&resp); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
				}

				return
			}

			http.Error(w, fmt.Sprintf("failed to initialize model: %v", err), http.StatusInternalServerError)
			return
		}
	}

	mem := s.model.Backend().BackendMemory()

	switch req.Operation {
	case llm.LoadOperationFit:
		// LoadOperationFit kann nicht fuer anderes verwendet werden
		s.closeModel()

	// LoadOperationAlloc soll fuer weitere Operationen offen bleiben

	case llm.LoadOperationCommit:
		s.status = llm.ServerStatusLoadingModel
		go s.loadModel()
	}

	resp := llm.LoadResponse{Success: true, Memory: mem}
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

// info ist der HTTP Handler fuer GPU-Device Informationen
func (s *Server) info(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	m := s.model

	if m == nil {
		startLoad := time.Now()

		// Dummy-Load um das Backend zu initialisieren
		f, err := os.CreateTemp("", "*.bin")
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to initialize backend: %v", err), http.StatusInternalServerError)
			return
		}
		defer f.Close()
		defer os.Remove(f.Name())

		if err := ggml.WriteGGUF(f, ggml.KV{
			"general.architecture": "llama",
			"tokenizer.ggml.model": "gpt2",
		}, nil); err != nil {
			http.Error(w, fmt.Sprintf("failed to initialize backend: %v", err), http.StatusInternalServerError)
			return
		}

		m, err = model.New(f.Name(), ml.BackendParams{NumThreads: runtime.NumCPU(), AllocMemory: false, GPULayers: ml.GPULayersList{{}}})
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to initialize backend: %v", err), http.StatusInternalServerError)
			return
		}
		slog.Debug("dummy model load took", "duration", time.Since(startLoad))
	}

	startDevices := time.Now()
	infos := m.Backend().BackendDevices()
	slog.Debug("gathering device infos took", "duration", time.Since(startDevices))
	if err := json.NewEncoder(w).Encode(&infos); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}
