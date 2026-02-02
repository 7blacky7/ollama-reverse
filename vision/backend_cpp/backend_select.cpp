/**
 * MODUL: backend_select.cpp
 * ZWECK: Zentrale Backend-Auswahl fuer Vision Encoder
 * INPUT: Praeferenz-String, GPU-ID
 * OUTPUT: ggml_backend_t
 * NEBENEFFEKTE: Backend-Initialisierung
 * ABHAENGIGKEITEN: ggml-backend.h, ggml-cuda.h (optional), ggml-metal.h (optional)
 * HINWEISE: Kann von allen Vision Encodern genutzt werden
 */

#include "backend_detect.h"

// GGML Headers
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Bedingte GPU-Backend Headers
#if defined(GGML_USE_CUDA) || defined(GGML_USE_CUBLAS)
    #include "ggml-cuda.h"
    #define HAS_CUDA_BACKEND 1
#else
    #define HAS_CUDA_BACKEND 0
#endif

#if defined(__APPLE__) && defined(GGML_USE_METAL)
    #include "ggml-metal.h"
    #define HAS_METAL_BACKEND 1
#else
    #define HAS_METAL_BACKEND 0
#endif

#include <cstring>

extern "C" {

/* ============================================================================
 * Backend-Erstellung
 * ============================================================================ */

/**
 * Erstellt ein GGML-Backend basierend auf Praeferenz.
 * Faellt auf CPU zurueck wenn gewuenschtes Backend nicht verfuegbar.
 */
ggml_backend_t backend_create_ggml(const char* preference, int gpu_id, int n_threads) {
    ggml_backend_t backend = nullptr;

    /* CUDA Backend versuchen */
#if HAS_CUDA_BACKEND
    if (strcmp(preference, "cuda") == 0 || strcmp(preference, "auto") == 0) {
        if (backend_cuda_available()) {
            backend = ggml_backend_cuda_init(gpu_id);
            if (backend) return backend;
        }
    }
#endif

    /* Metal Backend versuchen */
#if HAS_METAL_BACKEND
    if (strcmp(preference, "metal") == 0 || strcmp(preference, "auto") == 0) {
        if (backend_metal_available()) {
            backend = ggml_backend_metal_init();
            if (backend) return backend;
        }
    }
#endif

    /* CPU Fallback */
    backend = ggml_backend_cpu_init();
    if (backend && n_threads > 0) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    return backend;
}

/**
 * Erstellt das beste verfuegbare Backend automatisch.
 */
ggml_backend_t backend_create_best(int gpu_id, int n_threads) {
    return backend_create_ggml("auto", gpu_id, n_threads);
}

/**
 * Gibt den Namen des verwendeten Backends zurueck.
 */
const char* backend_get_ggml_name(ggml_backend_t backend) {
    if (!backend) return "none";
    return ggml_backend_name(backend);
}

/**
 * Prueft ob ein Backend GPU-beschleunigt ist.
 */
bool backend_is_gpu(ggml_backend_t backend) {
    if (!backend) return false;

    const char* name = ggml_backend_name(backend);
    if (!name) return false;

    /* Bekannte GPU-Backend Namen */
    if (strstr(name, "CUDA") != nullptr) return true;
    if (strstr(name, "Metal") != nullptr) return true;
    if (strstr(name, "Vulkan") != nullptr) return true;
    if (strstr(name, "SYCL") != nullptr) return true;

    return false;
}

/* ============================================================================
 * Buffer-Erstellung
 * ============================================================================ */

/**
 * Erstellt einen Buffer fuer das Backend.
 */
ggml_backend_buffer_t backend_create_buffer(ggml_backend_t backend, size_t size) {
    if (!backend || size == 0) return nullptr;

    /* Buffer-Typ vom Backend holen */
    ggml_backend_buffer_type_t buf_type = ggml_backend_get_default_buffer_type(backend);
    if (!buf_type) return nullptr;

    return ggml_backend_buft_alloc_buffer(buf_type, size);
}

/**
 * Gibt die empfohlene Buffer-Groesse fuer ein Modell zurueck.
 * Beruecksichtigt Alignment-Anforderungen des Backends.
 */
size_t backend_get_buffer_size(ggml_backend_t backend, size_t model_size) {
    if (!backend || model_size == 0) return 0;

    /* Buffer-Typ holen */
    ggml_backend_buffer_type_t buf_type = ggml_backend_get_default_buffer_type(backend);
    if (!buf_type) return model_size;

    /* Alignment beruecksichtigen */
    size_t alignment = ggml_backend_buft_get_alignment(buf_type);
    if (alignment > 1) {
        model_size = ((model_size + alignment - 1) / alignment) * alignment;
    }

    return model_size;
}

} /* extern "C" */
