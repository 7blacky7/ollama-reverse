// Package llm - Server Types und Interfaces
//
// Definiert die grundlegenden Typen und Interfaces für den LLM Server:
// - LlamaServer Interface mit allen öffentlichen Methoden
// - llmServer Basisstruct für gemeinsame Funktionalität
// - llamaServer und ollamaServer für spezifische Implementierungen
// - filteredEnv für sichere Umgebungsvariablen-Logging
package llm

import (
	"context"
	"log/slog"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// filteredEnv filtert Umgebungsvariablen für sicheres Logging
type filteredEnv []string

func (e filteredEnv) LogValue() slog.Value {
	var attrs []slog.Attr
	for _, env := range e {
		if key, value, ok := strings.Cut(env, "="); ok {
			switch {
			case strings.HasPrefix(key, "OLLAMA_"),
				strings.HasPrefix(key, "CUDA_"),
				strings.HasPrefix(key, "ROCR_"),
				strings.HasPrefix(key, "ROCM_"),
				strings.HasPrefix(key, "HIP_"),
				strings.HasPrefix(key, "GPU_"),
				strings.HasPrefix(key, "HSA_"),
				strings.HasPrefix(key, "GGML_"),
				slices.Contains([]string{
					"PATH",
					"LD_LIBRARY_PATH",
					"DYLD_LIBRARY_PATH",
				}, key):
				attrs = append(attrs, slog.String(key, value))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

// LlamaServer definiert das Interface für LLM Server Operationen
type LlamaServer interface {
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, input string) ([]float32, int, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	VRAMSize() uint64 // Total VRAM across all GPUs
	TotalSize() uint64
	VRAMByGPU(id ml.DeviceID) uint64
	Pid() int
	GetPort() int
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
	HasExited() bool
}

// llmServer ist eine Instanz eines Runners für ein einzelnes Model
type llmServer struct {
	port      int
	cmd       *exec.Cmd
	done      chan error // Channel signalisiert wenn Prozess beendet
	status    *StatusWriter
	options   api.Options
	modelPath string

	loadRequest LoadRequest       // Parameter für Runner-Initialisierung
	mem         *ml.BackendMemory // Speicherallokationen für dieses Model

	// llamaModel ist eine CGO llama.cpp Model Definition
	// nil wenn der neue Engine läuft
	llamaModel     *llama.Model
	llamaModelLock *sync.Mutex

	totalLayers  uint64
	loadStart    time.Time // Ladezeit des Models
	loadProgress float32

	sem *semaphore.Weighted
}

// llamaServer erweitert llmServer für llama.cpp Backend
type llamaServer struct {
	llmServer

	ggml *ggml.GGML
}

// ollamaServer erweitert llmServer für Ollama Engine
type ollamaServer struct {
	llmServer

	textProcessor model.TextProcessor // Text Encoding/Decoding
}

// ModelPath gibt den Pfad zum geladenen Model zurück
func (s *llmServer) ModelPath() string {
	return s.modelPath
}
