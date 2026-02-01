// Package llm - Model Loading (llamaServer)
//
// Hauptdatei für Model-Loading:
// - LoadOperation Enum für verschiedene Lade-Stufen
// - LoadRequest/LoadResponse für Kommunikation mit Runner
// - llamaServer.Load Implementierung
package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/ml"
)

// LoadOperation definiert die Stufen des Model-Ladens
type LoadOperation int

const (
	LoadOperationFit    LoadOperation = iota // Speicherbedarf berechnen
	LoadOperationAlloc                       // Speicher allokieren
	LoadOperationCommit                      // Weights laden (final)
	LoadOperationClose                       // Model schließen
)

func (o LoadOperation) String() string {
	switch o {
	case LoadOperationFit:
		return "fit"
	case LoadOperationAlloc:
		return "alloc"
	case LoadOperationCommit:
		return "commit"
	case LoadOperationClose:
		return "close"
	default:
		return "unknown"
	}
}

// LoadRequest enthält Parameter für das Model-Laden
type LoadRequest struct {
	Operation LoadOperation

	LoraPath       []string
	Parallel       int
	BatchSize      int
	FlashAttention ml.FlashAttentionType
	KvSize         int
	KvCacheType    string
	NumThreads     int
	GPULayers      ml.GPULayersList
	MultiUserCache bool

	// Legacy Felder - nicht für Ollama Engine
	ProjectorPath string
	MainGPU       int
	UseMmap       bool
}

// LoadResponse enthält das Ergebnis des Model-Ladens
type LoadResponse struct {
	Success bool
	Memory  ml.BackendMemory
}

// ErrLoadRequiredFull signalisiert dass GPU zu wenig Speicher hat
var ErrLoadRequiredFull = errors.New("unable to load full model on GPU")

// Load lädt das Model auf die GPUs (llamaServer Variante)
func (s *llamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	slog.Info("loading model", "model layers", s.totalLayers, "requested", s.options.NumGPU)

	gpus := append(make([]ml.DeviceInfo, 0, len(systemGPUs)), systemGPUs...)
	s.initializeMemoryLayout(gpus)

	// Embedding Model Check
	_, isEmbedding := s.ggml.KV()[fmt.Sprintf("%s.pooling_type", s.ggml.KV().Architecture())]
	if isEmbedding && s.loadRequest.BatchSize < s.options.NumCtx {
		s.loadRequest.BatchSize = s.options.NumCtx
		slog.Info("embedding model detected, setting batch size to context length", "batch_size", s.loadRequest.BatchSize)
	}

	kv, graphPartialOffload, graphFullOffload := s.ggml.GraphSize(
		uint64(s.options.NumCtx), uint64(s.loadRequest.BatchSize),
		s.loadRequest.Parallel, s.loadRequest.KvCacheType, s.loadRequest.FlashAttention)

	s.assignLayersToCPU(kv, gpus)
	projectorGPU := s.handleProjector(gpus)
	graphPartialOffload, graphFullOffload = s.adjustGraphSizes(kv, graphPartialOffload, graphFullOffload, gpus)

	gpuLayers, err := s.iterateLayouts(systemInfo, gpus, requireFull, graphPartialOffload, graphFullOffload)
	if err != nil {
		return nil, err
	}

	s.finalizeMemoryLayout(gpuLayers, projectorGPU, graphPartialOffload, graphFullOffload)
	s.configureMmap(gpus, systemInfo)

	if err := s.waitUntilRunnerLaunched(ctx); err != nil {
		return nil, err
	}

	s.loadRequest.GPULayers = gpuLayers
	resp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)
	if err != nil {
		return nil, err
	}

	if !resp.Success {
		return nil, errors.New("failed to allocate memory for model")
	}

	return uniqueDeviceIDs(s.loadRequest.GPULayers), s.WaitUntilRunning(ctx)
}
