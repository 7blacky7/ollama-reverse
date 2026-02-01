// Package llm - Server Erstellung
//
// Funktionen zum Erstellen von LLM Servern:
// - LoadModel: GGML Model von Disk laden
// - NewLlamaServer: Server für gegebene GPUs erstellen
// - Konfiguration von Flash Attention und KV Cache
package llm

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// LoadModel lädt ein Model von Disk im GGML Format.
// maxArraySize begrenzt die Größe gesammelter Arrays (Standard: 1024, negativ: alle)
func LoadModel(modelPath string, maxArraySize int) (*ggml.GGML, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, err
	}

	f, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := ggml.Decode(f, maxArraySize)
	return ggml, err
}

// NewLlamaServer erstellt einen Server für die gegebenen GPUs
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	llamaModel, textProcessor, err := initializeModel(modelPath, f, projectors)
	if err != nil {
		return nil, err
	}

	// Context-Größe validieren
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}
	opts.NumBatch = min(opts.NumBatch, opts.NumCtx)

	loadRequest := buildLoadRequest(systemInfo, opts, adapters, projectors, numParallel, llamaModel)
	loadRequest = configureFlashAttention(loadRequest, f, gpus, textProcessor)

	s, err := startServerProcess(modelPath, opts, loadRequest, llamaModel, gpus, f, numParallel)
	if err != nil {
		return nil, err
	}

	if textProcessor != nil {
		return &ollamaServer{llmServer: *s, textProcessor: textProcessor}, nil
	}
	return &llamaServer{llmServer: *s, ggml: f}, nil
}

// initializeModel lädt das Model und wählt die Engine
func initializeModel(modelPath string, f *ggml.GGML, projectors []string) (*llama.Model, model.TextProcessor, error) {
	var llamaModel *llama.Model
	var textProcessor model.TextProcessor
	var err error

	if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
		if len(projectors) == 0 {
			textProcessor, err = model.NewTextProcessor(modelPath)
		} else {
			err = errors.New("split vision models aren't supported")
		}
		if err != nil {
			slog.Debug("model not yet supported by Ollama engine, switching to compatibility mode", "model", modelPath, "error", err)
		}
	}

	if textProcessor == nil {
		llamaModel, err = llama.LoadModelFromFile(modelPath, llama.ModelParams{VocabOnly: true})
		if err != nil {
			return nil, nil, err
		}
	}

	return llamaModel, textProcessor, nil
}

// buildLoadRequest erstellt die LoadRequest Struktur
func buildLoadRequest(systemInfo ml.SystemInfo, opts api.Options, adapters, projectors []string, numParallel int, llamaModel *llama.Model) LoadRequest {
	req := LoadRequest{
		LoraPath:       adapters,
		KvSize:         opts.NumCtx * numParallel,
		BatchSize:      opts.NumBatch,
		Parallel:       numParallel,
		MultiUserCache: envconfig.MultiUserCache(),
	}

	if opts.NumThread > 0 {
		req.NumThreads = opts.NumThread
	} else if systemInfo.ThreadCount > 0 {
		req.NumThreads = systemInfo.ThreadCount
	}

	if opts.MainGPU > 0 {
		req.MainGPU = opts.MainGPU
	}

	if len(projectors) > 0 && llamaModel != nil {
		req.ProjectorPath = projectors[0]
	}

	return req
}

// configureFlashAttention konfiguriert Flash Attention und KV Cache
func configureFlashAttention(req LoadRequest, f *ggml.GGML, gpus []ml.DeviceInfo, textProcessor model.TextProcessor) LoadRequest {
	faUserSet := envconfig.FlashAttention(true) == envconfig.FlashAttention(false)
	fa := envconfig.FlashAttention(f.FlashAttention())

	if fa && !ml.FlashAttentionSupported(gpus) {
		slog.Warn("flash attention enabled but not supported by gpu")
		fa = false
	}

	if fa && !f.SupportsFlashAttention() {
		slog.Warn("flash attention enabled but not supported by model")
		fa = false
	}

	kvct := strings.ToLower(envconfig.KvCacheType())

	if textProcessor == nil {
		req.FlashAttention = determineFlashAttentionType(faUserSet, fa)
		req.KvCacheType = configureKvCache(f, kvct, req.FlashAttention)
	} else {
		if fa {
			slog.Info("enabling flash attention")
			req.FlashAttention = ml.FlashAttentionEnabled
			if f.SupportsKVCacheType(kvct) {
				req.KvCacheType = kvct
			} else {
				slog.Warn("kv cache type not supported by model", "type", kvct)
			}
		} else if kvct != "" && kvct != "f16" {
			slog.Warn("quantized kv cache requested but flash attention disabled", "type", kvct)
		}
	}

	return req
}

func determineFlashAttentionType(userSet, enabled bool) ml.FlashAttentionType {
	if !userSet {
		return ml.FlashAttentionAuto
	}
	if enabled {
		return ml.FlashAttentionEnabled
	}
	return ml.FlashAttentionDisabled
}

func configureKvCache(f *ggml.GGML, kvct string, fa ml.FlashAttentionType) string {
	if kvct == "" {
		return ""
	}

	if f.KVCacheTypeIsQuantized(kvct) {
		if fa != ml.FlashAttentionEnabled {
			slog.Warn("OLLAMA_FLASH_ATTENTION must be enabled to use a quantized OLLAMA_KV_CACHE_TYPE", "type", kvct)
			return ""
		}
		if f.SupportsKVCacheType(kvct) {
			return kvct
		}
		slog.Warn("unsupported OLLAMA_KV_CACHE_TYPE", "type", kvct)
		return ""
	}

	if f.SupportsKVCacheType(kvct) {
		return kvct
	}
	slog.Warn("unsupported OLLAMA_KV_CACHE_TYPE", "type", kvct)
	return ""
}

// startServerProcess startet den Server-Prozess
func startServerProcess(modelPath string, opts api.Options, loadRequest LoadRequest, llamaModel *llama.Model, gpus []ml.DeviceInfo, f *ggml.GGML, numParallel int) (*llmServer, error) {
	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)
	cmd, port, err := StartRunner(
		llamaModel == nil,
		modelPath,
		gpuLibs,
		status,
		ml.GetVisibleDevicesEnv(gpus, false),
	)

	s := &llmServer{
		port:           port,
		cmd:            cmd,
		status:         status,
		options:        opts,
		modelPath:      modelPath,
		loadRequest:    loadRequest,
		llamaModel:     llamaModel,
		llamaModelLock: &sync.Mutex{},
		sem:            semaphore.NewWeighted(int64(numParallel)),
		totalLayers:    f.KV().BlockCount() + 1,
		loadStart:      time.Now(),
		done:           make(chan error, 1),
	}

	if err != nil {
		return nil, handleStartError(err, s, llamaModel)
	}

	go monitorProcess(s)
	return s, nil
}

func handleStartError(err error, s *llmServer, llamaModel *llama.Model) error {
	var msg string
	if s.status != nil && s.status.LastErrMsg != "" {
		msg = s.status.LastErrMsg
	}
	if llamaModel != nil {
		llama.FreeModel(llamaModel)
	}
	return fmt.Errorf("error starting runner: %v %s", err, msg)
}

func monitorProcess(s *llmServer) {
	err := s.cmd.Wait()
	if err != nil && s.status != nil && s.status.LastErrMsg != "" {
		slog.Error("llama runner terminated", "error", err)
		if strings.Contains(s.status.LastErrMsg, "unknown model") {
			s.status.LastErrMsg = "this model is not supported by your version of Ollama. You may need to upgrade"
		}
		s.done <- errors.New(s.status.LastErrMsg)
	} else {
		s.done <- err
	}
}
