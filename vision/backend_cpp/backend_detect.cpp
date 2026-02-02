/**
 * MODUL: backend_detect.cpp
 * ZWECK: Implementierung der Backend-Erkennung fuer CUDA und Metal
 * INPUT: Device-IDs
 * OUTPUT: Device-Informationen via backend_device_info
 * NEBENEFFEKTE: Hardware-Abfragen, optionale Initialisierung
 * ABHAENGIGKEITEN: CUDA Runtime (optional), Metal Framework (optional)
 * HINWEISE: Bedingte Kompilierung via Praeprozessor-Makros
 */

#include "backend_detect.h"

#include <cstring>
#include <cstdio>

/* ============================================================================
 * Plattform-Detection
 * ============================================================================ */

/* CUDA verfuegbar wenn GGML_USE_CUDA definiert */
#if defined(GGML_USE_CUDA) || defined(GGML_USE_CUBLAS)
    #define HAS_CUDA 1
    #include <cuda_runtime.h>
#else
    #define HAS_CUDA 0
#endif

/* Metal verfuegbar auf Apple-Plattformen */
#if defined(__APPLE__) && defined(GGML_USE_METAL)
    #define HAS_METAL 1
    #include <TargetConditionals.h>
    // Metal-Funktionen werden extern deklariert (Obj-C++ Wrapper)
    extern "C" {
        bool metal_is_available(void);
        int metal_get_device_count(void);
        int metal_get_device_info(int device_id, char* name, size_t name_len,
                                  uint64_t* memory_total, uint64_t* memory_free);
        uint64_t metal_get_recommended_working_set(int device_id);
    }
#else
    #define HAS_METAL 0
#endif

/* ============================================================================
 * CUDA Implementierung
 * ============================================================================ */

extern "C" {

bool backend_cuda_available(void) {
#if HAS_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

int backend_cuda_device_count(void) {
#if HAS_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
#else
    return 0;
#endif
}

int backend_cuda_get_device(int device_id, backend_device_info* info) {
    if (!info) return -1;

    /* Struktur initialisieren */
    memset(info, 0, sizeof(backend_device_info));
    info->device_id = device_id;

#if HAS_CUDA
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) return -2;

    /* Name kopieren */
    strncpy(info->name, props.name, BACKEND_MAX_NAME_LEN - 1);
    info->name[BACKEND_MAX_NAME_LEN - 1] = '\0';

    /* Speicher abfragen */
    info->memory_total = props.totalGlobalMem;

    /* Freien Speicher ermitteln (erfordert Geraetewechsel) */
    int current_device;
    cudaGetDevice(&current_device);
    if (cudaSetDevice(device_id) == cudaSuccess) {
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            info->memory_free = free_mem;
        }
        cudaSetDevice(current_device);
    }

    /* Compute Capability */
    info->compute_major = props.major;
    info->compute_minor = props.minor;

    return 0;
#else
    strncpy(info->name, "CUDA not available", BACKEND_MAX_NAME_LEN - 1);
    return -3;
#endif
}

int backend_cuda_set_device(int device_id) {
#if HAS_CUDA
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

uint64_t backend_cuda_get_free_memory(int device_id) {
#if HAS_CUDA
    int current_device;
    cudaGetDevice(&current_device);

    if (cudaSetDevice(device_id) != cudaSuccess) return 0;

    size_t free_mem, total_mem;
    uint64_t result = 0;

    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        result = free_mem;
    }

    cudaSetDevice(current_device);
    return result;
#else
    return 0;
#endif
}

/* ============================================================================
 * Metal Implementierung
 * ============================================================================ */

bool backend_metal_available(void) {
#if HAS_METAL
    return metal_is_available();
#else
    return false;
#endif
}

int backend_metal_device_count(void) {
#if HAS_METAL
    return metal_get_device_count();
#else
    return 0;
#endif
}

int backend_metal_get_device(int device_id, backend_device_info* info) {
    if (!info) return -1;

    /* Struktur initialisieren */
    memset(info, 0, sizeof(backend_device_info));
    info->device_id = device_id;

#if HAS_METAL
    uint64_t mem_total = 0, mem_free = 0;
    int result = metal_get_device_info(device_id, info->name,
                                       BACKEND_MAX_NAME_LEN,
                                       &mem_total, &mem_free);
    if (result != 0) return result;

    info->memory_total = mem_total;
    info->memory_free = mem_free;

    /* Metal hat keine Compute Capability */
    info->compute_major = 0;
    info->compute_minor = 0;

    return 0;
#else
    strncpy(info->name, "Metal not available", BACKEND_MAX_NAME_LEN - 1);
    return -3;
#endif
}

uint64_t backend_metal_get_recommended_memory(int device_id) {
#if HAS_METAL
    return metal_get_recommended_working_set(device_id);
#else
    return 0;
#endif
}

/* ============================================================================
 * Utility Funktionen
 * ============================================================================ */

const char* backend_get_best(void) {
    if (backend_cuda_available()) return "cuda";
    if (backend_metal_available()) return "metal";
    return "cpu";
}

void backend_init_all(void) {
#if HAS_CUDA
    /* CUDA Context initialisieren */
    if (backend_cuda_available()) {
        cudaFree(0);  // Dummy-Aufruf zur Initialisierung
    }
#endif

#if HAS_METAL
    /* Metal initialisieren (falls noetig) */
    backend_metal_available();
#endif
}

} /* extern "C" */
