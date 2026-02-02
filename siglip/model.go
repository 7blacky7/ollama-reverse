//go:build siglip
// ============================================================================
// MODUL: model
// ZWECK: SigLIP Model-Verwaltung und Initialisierung
// INPUT: GGUF-Modell-Pfad, Konfigurationsoptionen
// OUTPUT: Geladenes Model, Embeddings
// NEBENEFFEKTE: Laedt nativen Code, alloziert Speicher
// ABHAENGIGKEITEN: siglip.h (C-Library), types.go
// HINWEISE: CGO Flags sind NUR hier definiert
// ============================================================================

package siglip

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo CFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include
#cgo CFLAGS: -I${SRCDIR}/../llama.cpp.upstream/ggml/include
#cgo LDFLAGS: -lm -lstdc++
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lpthread

#include <stdlib.h>
#include "siglip.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// ============================================================================
// Model Struct
// ============================================================================

// Model repraesentiert ein geladenes SigLIP-Modell.
// Thread-safe durch internen Mutex.
type Model struct {
	ctx    *C.struct_siglip_ctx // C-Kontext
	mu     sync.Mutex           // Thread-Safety
	closed bool                 // Status-Flag

	// Gecachte Metadaten
	embeddingDim int       // Embedding-Dimension
	imageSize    int       // Erwartete Bildgroesse
	modelType    ModelType // Modell-Variante
	modelName    string    // Modell-Name
}

// ============================================================================
// LoadModel - Modell aus GGUF-Datei laden
// ============================================================================

// LoadModel laedt ein SigLIP-Modell aus einer GGUF-Datei.
// Optionale Konfiguration ueber Functional Options.
func LoadModel(path string, opts ...Option) (*Model, error) {
	// Standard-Optionen mit uebergebenen Optionen zusammenfuehren
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

	// Model-Instanz erstellen
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

// ============================================================================
// Close - Modell freigeben
// ============================================================================

// Close gibt das Modell frei und setzt alle Ressourcen zurueck.
// Thread-safe, kann mehrfach aufgerufen werden.
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

// ============================================================================
// Getter-Methoden
// ============================================================================

// EmbeddingDim gibt die Embedding-Dimension zurueck.
func (m *Model) EmbeddingDim() int {
	return m.embeddingDim
}

// ImageSize gibt die erwartete Bildgroesse zurueck.
func (m *Model) ImageSize() int {
	return m.imageSize
}

// ModelType gibt den Modell-Typ zurueck.
func (m *Model) ModelType() ModelType {
	return m.modelType
}

// ModelName gibt den Modell-Namen zurueck.
func (m *Model) ModelName() string {
	return m.modelName
}

// ============================================================================
// Functional Options
// ============================================================================

// Option ist eine funktionale Option fuer LoadModel.
type Option func(*options)

// options speichert die Modell-Konfiguration.
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

// defaultOptions gibt die Standard-Konfiguration zurueck.
func defaultOptions() *options {
	const (
		defaultGPULayers = -1 // Alle Layer auf GPU
		defaultMainGPU   = 0
		defaultBatchSize = 1
	)

	return &options{
		backend:     BackendCPU,
		logLevel:    LogInfo,
		embedFormat: EmbedF32,
		nThreads:    runtime.NumCPU(),
		nGPULayers:  defaultGPULayers,
		mainGPU:     defaultMainGPU,
		useMmap:     true,
		useMlock:    false,
		batchSize:   defaultBatchSize,
	}
}

// WithBackend setzt das Compute-Backend.
func WithBackend(backend Backend) Option {
	return func(o *options) {
		o.backend = backend
	}
}

// WithLogLevel setzt das Log-Level.
func WithLogLevel(level LogLevel) Option {
	return func(o *options) {
		o.logLevel = level
	}
}

// WithEmbedFormat setzt das Embedding-Format.
func WithEmbedFormat(format EmbedFormat) Option {
	return func(o *options) {
		o.embedFormat = format
	}
}

// WithThreads setzt die Anzahl der CPU-Threads.
func WithThreads(n int) Option {
	return func(o *options) {
		o.nThreads = n
	}
}

// WithGPULayers setzt die Anzahl der GPU-Layers (-1 fuer alle).
func WithGPULayers(n int) Option {
	return func(o *options) {
		o.nGPULayers = n
	}
}

// WithMainGPU setzt den Haupt-GPU Index.
func WithMainGPU(gpu int) Option {
	return func(o *options) {
		o.mainGPU = gpu
	}
}

// WithMmap aktiviert/deaktiviert Memory-Mapping.
func WithMmap(enabled bool) Option {
	return func(o *options) {
		o.useMmap = enabled
	}
}

// WithMlock aktiviert/deaktiviert Memory-Locking.
func WithMlock(enabled bool) Option {
	return func(o *options) {
		o.useMlock = enabled
	}
}

// WithBatchSize setzt die Batch-Groesse.
func WithBatchSize(size int) Option {
	return func(o *options) {
		o.batchSize = size
	}
}
