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
//
// Diese Datei enthaelt:
// - Model Struct: Repraesentiert ein geladenes SigLIP-Modell
// - LoadModel: Laedt ein Modell aus einer GGUF-Datei
// - Close: Gibt das Modell frei
// - Getter-Methoden: EmbeddingDim, ImageSize, ModelType, ModelName
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
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// ============================================================================
// Model Struct
// ============================================================================

// Model repraesentiert ein geladenes SigLIP-Modell.
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

// ============================================================================
// LoadModel
// ============================================================================

// LoadModel laedt ein SigLIP-Modell aus einer GGUF-Datei.
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

// ============================================================================
// Close
// ============================================================================

// Close gibt das Modell frei.
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

// Hinweis: Global Functions (Version, BuildInfo, SystemInfo, SetLogLevel,
// GetLastError, ClearError, BackendAvailable, AvailableBackends) sind in utils.go
