// Package siglip provides Go bindings for the SigLIP Vision Encoder.
//
// SigLIP (Sigmoid Loss for Language Image Pre-Training) ist ein Vision Transformer
// zur Generierung von Image-Embeddings. Diese Bindings wrappen die C-Implementation
// aus llama.cpp/src/siglip.h.
//
// Verwendung:
//
//	model, err := siglip.LoadModel("siglip-vit-b.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	embedding, err := model.Encode(imageData)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	floats := embedding.ToFloat32()
package siglip

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/src
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build -lsiglip -lggml -lm -lstdc++
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lpthread

#include <stdlib.h>
#include "siglip.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Fehler-Definitionen
var (
	ErrModelNotLoaded    = errors.New("siglip: model not loaded")
	ErrInvalidImage      = errors.New("siglip: invalid image data")
	ErrEncodingFailed    = errors.New("siglip: encoding failed")
	ErrInvalidParameters = errors.New("siglip: invalid parameters")
	ErrDimensionMismatch = errors.New("siglip: embedding dimension mismatch")
)

// ============================================================================
// Enums
// ============================================================================

// ModelType definiert den Modell-Typ
type ModelType int

const (
	ModelVitB16    ModelType = C.SIGLIP_MODEL_VIT_B_16   // ViT-Base, Patch 16, 86M params
	ModelVitL16    ModelType = C.SIGLIP_MODEL_VIT_L_16   // ViT-Large, Patch 16, 303M params
	ModelVitSO400M ModelType = C.SIGLIP_MODEL_VIT_SO400M // ViT-SO400M, Patch 14, 400M params
	ModelUnknown   ModelType = C.SIGLIP_MODEL_UNKNOWN
)

// String gibt den Namen des Modell-Typs zurueck
func (t ModelType) String() string {
	switch t {
	case ModelVitB16:
		return "ViT-B/16"
	case ModelVitL16:
		return "ViT-L/16"
	case ModelVitSO400M:
		return "ViT-SO400M"
	default:
		return "Unknown"
	}
}

// Backend definiert das Compute-Backend
type Backend int

const (
	BackendCPU    Backend = C.SIGLIP_BACKEND_CPU    // CPU (GGML)
	BackendCUDA   Backend = C.SIGLIP_BACKEND_CUDA   // NVIDIA CUDA
	BackendMetal  Backend = C.SIGLIP_BACKEND_METAL  // Apple Metal
	BackendVulkan Backend = C.SIGLIP_BACKEND_VULKAN // Vulkan (experimentell)
)

// String gibt den Namen des Backends zurueck
func (b Backend) String() string {
	switch b {
	case BackendCPU:
		return "CPU"
	case BackendCUDA:
		return "CUDA"
	case BackendMetal:
		return "Metal"
	case BackendVulkan:
		return "Vulkan"
	default:
		return "Unknown"
	}
}

// LogLevel definiert das Log-Level
type LogLevel int

const (
	LogNone  LogLevel = C.SIGLIP_LOG_NONE
	LogError LogLevel = C.SIGLIP_LOG_ERROR
	LogWarn  LogLevel = C.SIGLIP_LOG_WARN
	LogInfo  LogLevel = C.SIGLIP_LOG_INFO
	LogDebug LogLevel = C.SIGLIP_LOG_DEBUG
)

// EmbedFormat definiert das Embedding-Format
type EmbedFormat int

const (
	EmbedF32        EmbedFormat = C.SIGLIP_EMBED_F32        // float32 Array
	EmbedF16        EmbedFormat = C.SIGLIP_EMBED_F16        // float16 Array
	EmbedNormalized EmbedFormat = C.SIGLIP_EMBED_NORMALIZED // L2-normalisiert
)

// ============================================================================
// Options
// ============================================================================

// Option ist eine funktionale Option fuer LoadModel
type Option func(*options)

type options struct {
	backend     Backend
	logLevel    LogLevel
	embedFormat EmbedFormat
	nThreads    int
	nGPULayers  int
	mainGPU     int
	useMmap     bool
	useMlock    bool
	batchSize   int
}

func defaultOptions() *options {
	return &options{
		backend:     BackendCPU,
		logLevel:    LogInfo,
		embedFormat: EmbedF32,
		nThreads:    runtime.NumCPU(),
		nGPULayers:  -1, // alle
		mainGPU:     0,
		useMmap:     true,
		useMlock:    false,
		batchSize:   1,
	}
}

// WithBackend setzt das Compute-Backend
func WithBackend(backend Backend) Option {
	return func(o *options) {
		o.backend = backend
	}
}

// WithLogLevel setzt das Log-Level
func WithLogLevel(level LogLevel) Option {
	return func(o *options) {
		o.logLevel = level
	}
}

// WithEmbedFormat setzt das Embedding-Format
func WithEmbedFormat(format EmbedFormat) Option {
	return func(o *options) {
		o.embedFormat = format
	}
}

// WithThreads setzt die Anzahl der CPU-Threads
func WithThreads(n int) Option {
	return func(o *options) {
		o.nThreads = n
	}
}

// WithGPULayers setzt die Anzahl der GPU-Layers (-1 fuer alle)
func WithGPULayers(n int) Option {
	return func(o *options) {
		o.nGPULayers = n
	}
}

// WithMainGPU setzt den Haupt-GPU Index
func WithMainGPU(gpu int) Option {
	return func(o *options) {
		o.mainGPU = gpu
	}
}

// WithMmap aktiviert/deaktiviert Memory-Mapping
func WithMmap(enabled bool) Option {
	return func(o *options) {
		o.useMmap = enabled
	}
}

// WithMlock aktiviert/deaktiviert Memory-Locking
func WithMlock(enabled bool) Option {
	return func(o *options) {
		o.useMlock = enabled
	}
}

// WithBatchSize setzt die Batch-Groesse
func WithBatchSize(size int) Option {
	return func(o *options) {
		o.batchSize = size
	}
}

// ============================================================================
// Model
// ============================================================================

// Model repraesentiert ein geladenes SigLIP-Modell
type Model struct {
	ctx    *C.struct_siglip_ctx
	mu     sync.Mutex
	closed bool

	// Cached Info
	embeddingDim int
	imageSize    int
	modelType    ModelType
	modelName    string
}

// LoadModel laedt ein SigLIP-Modell aus einer GGUF-Datei
func LoadModel(path string, opts ...Option) (*Model, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(o)
	}

	// C-Parameter erstellen
	var params C.struct_siglip_params
	params.backend = C.enum_siglip_backend(o.backend)
	params.log_level = C.enum_siglip_log_level(o.logLevel)
	params.embed_format = C.enum_siglip_embed_format(o.embedFormat)
	params.n_threads = C.int(o.nThreads)
	params.n_gpu_layers = C.int(o.nGPULayers)
	params.main_gpu = C.int(o.mainGPU)
	params.use_mmap = C.bool(o.useMmap)
	params.use_mlock = C.bool(o.useMlock)
	params.batch_size = C.int(o.batchSize)

	// Pfad zu C-String konvertieren
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	// Modell laden
	ctx := C.siglip_load_model(cPath, params)
	if ctx == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrModelNotLoaded
	}

	// Model erstellen
	m := &Model{
		ctx:          ctx,
		embeddingDim: int(C.siglip_get_embedding_dim(ctx)),
		imageSize:    int(C.siglip_get_image_size(ctx)),
		modelType:    ModelType(C.siglip_get_model_type(ctx)),
	}

	// Model-Name holen
	cName := C.siglip_get_model_name(ctx)
	if cName != nil {
		m.modelName = C.GoString(cName)
	}

	// Finalizer setzen fuer automatisches Cleanup
	runtime.SetFinalizer(m, (*Model).Close)

	return m, nil
}

// Close gibt das Modell frei
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}

	if m.ctx != nil {
		C.siglip_free(m.ctx)
		m.ctx = nil
	}

	m.closed = true
	runtime.SetFinalizer(m, nil)

	return nil
}

// EmbeddingDim gibt die Embedding-Dimension zurueck
func (m *Model) EmbeddingDim() int {
	return m.embeddingDim
}

// ImageSize gibt die erwartete Bildgroesse zurueck
func (m *Model) ImageSize() int {
	return m.imageSize
}

// ModelType gibt den Modell-Typ zurueck
func (m *Model) ModelType() ModelType {
	return m.modelType
}

// ModelName gibt den Modell-Namen zurueck
func (m *Model) ModelName() string {
	return m.modelName
}

// Global Functions (Version, BuildInfo, SystemInfo, SetLogLevel, GetLastError,
// ClearError, BackendAvailable, AvailableBackends) sind in utils.go
