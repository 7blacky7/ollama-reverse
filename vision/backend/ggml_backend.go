// MODUL: ggml_backend
// ZWECK: Integration mit GGML Backend-System fuer Vision Encoder
// INPUT: Backend-Typ, LoadOptions
// OUTPUT: ggml_backend_t Pointer via CGO
// NEBENEFFEKTE: GGML Backend Initialisierung
// ABHAENGIGKEITEN: ggml-backend.h (extern)
// HINWEISE: Zentrale Stelle fuer Backend-Auswahl aller Vision Encoder

package backend

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp.upstream/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../build -lggml

#include "ggml-backend.h"

// Bedingte CUDA/Metal Includes
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif

// Backend-Erstellung basierend auf Typ
ggml_backend_t create_backend(const char* backend_type, int gpu_id, int n_threads) {
    ggml_backend_t backend = NULL;

    // CUDA Backend
    #if defined(GGML_USE_CUDA)
    if (strcmp(backend_type, "cuda") == 0) {
        backend = ggml_backend_cuda_init(gpu_id);
        if (backend) return backend;
    }
    #endif

    // Metal Backend
    #if defined(GGML_USE_METAL)
    if (strcmp(backend_type, "metal") == 0) {
        backend = ggml_backend_metal_init();
        if (backend) return backend;
    }
    #endif

    // CPU Fallback
    backend = ggml_backend_cpu_init();
    if (backend && n_threads > 0) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    return backend;
}
*/
import "C"

import (
	"unsafe"
)

// ============================================================================
// GGML Backend Wrapper
// ============================================================================

// GGMLBackend kapselt einen ggml_backend_t Pointer.
type GGMLBackend struct {
	ptr     unsafe.Pointer
	backend Backend
	gpuID   int
}

// CreateGGMLBackend erstellt ein GGML-Backend basierend auf Typ.
func CreateGGMLBackend(b Backend, gpuID int, nThreads int) (*GGMLBackend, error) {
	cType := C.CString(string(b))
	defer C.free(unsafe.Pointer(cType))

	ptr := C.create_backend(cType, C.int(gpuID), C.int(nThreads))
	if ptr == nil {
		return nil, &BackendError{
			Backend: b,
			Op:      "create",
			Code:    -1,
		}
	}

	return &GGMLBackend{
		ptr:     unsafe.Pointer(ptr),
		backend: b,
		gpuID:   gpuID,
	}, nil
}

// CreateBestGGMLBackend erstellt das beste verfuegbare Backend.
func CreateBestGGMLBackend(gpuID int, nThreads int) (*GGMLBackend, error) {
	best := SelectBestBackend()
	return CreateGGMLBackend(best, gpuID, nThreads)
}

// Pointer gibt den rohen ggml_backend_t Pointer zurueck.
func (g *GGMLBackend) Pointer() unsafe.Pointer {
	return g.ptr
}

// Backend gibt den Backend-Typ zurueck.
func (g *GGMLBackend) Backend() Backend {
	return g.backend
}

// GPUID gibt den GPU-Index zurueck.
func (g *GGMLBackend) GPUID() int {
	return g.gpuID
}

// Close gibt das Backend frei.
func (g *GGMLBackend) Close() {
	if g.ptr != nil {
		C.ggml_backend_free((*C.ggml_backend)(g.ptr))
		g.ptr = nil
	}
}
