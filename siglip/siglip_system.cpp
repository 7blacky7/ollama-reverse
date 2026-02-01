/**
 * siglip_system.cpp - System-Info, Backend-Detection, Version
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - Version und Build-Info
 * - Backend-Verfuegbarkeit pruefen
 * - System-Informationen (SIMD, etc.)
 * - Globale Log-Einstellungen
 */

#include "siglip.h"

#include <string>

// GGML Headers
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

// ============================================================================
// Version und Build-Info
// ============================================================================

const char * siglip_version(void) {
    return "0.1.0";
}

const char * siglip_build_info(void) {
    static std::string info;

    if (info.empty()) {
        info = "siglip built with:";

        #ifdef __GNUC__
        info += " GCC " + std::to_string(__GNUC__) + "." +
                std::to_string(__GNUC_MINOR__);
        #endif

        #ifdef _MSC_VER
        info += " MSVC " + std::to_string(_MSC_VER);
        #endif

        #ifdef GGML_USE_CUDA
        info += " CUDA";
        #endif

        #ifdef GGML_USE_METAL
        info += " Metal";
        #endif

        #ifdef __AVX2__
        info += " AVX2";
        #endif

        #ifdef __AVX512F__
        info += " AVX512";
        #endif

        #ifdef __ARM_NEON
        info += " NEON";
        #endif
    }

    return info.c_str();
}

// ============================================================================
// System-Informationen
// ============================================================================

const char * siglip_system_info(void) {
    static std::string info;

    if (info.empty()) {
        if (ggml_cpu_has_avx())    info += "AVX ";
        if (ggml_cpu_has_avx2())   info += "AVX2 ";
        if (ggml_cpu_has_avx512()) info += "AVX512 ";
        if (ggml_cpu_has_fma())    info += "FMA ";
        if (ggml_cpu_has_f16c())   info += "F16C ";

        #ifdef __ARM_NEON
        info += "NEON ";
        #endif

        if (info.empty()) {
            info = "No SIMD";
        }
    }

    return info.c_str();
}

// ============================================================================
// Backend-Verfuegbarkeit
// ============================================================================

bool siglip_backend_available(siglip_backend backend) {
    switch (backend) {
        case SIGLIP_BACKEND_CPU:
            return true;

        case SIGLIP_BACKEND_CUDA:
            #ifdef GGML_USE_CUDA
            return true;
            #else
            return false;
            #endif

        case SIGLIP_BACKEND_METAL:
            #ifdef GGML_USE_METAL
            return true;
            #else
            return false;
            #endif

        case SIGLIP_BACKEND_VULKAN:
            #ifdef GGML_USE_VULKAN
            return true;
            #else
            return false;
            #endif

        default:
            return false;
    }
}

int siglip_get_available_backends(siglip_backend * backends, int max_backends) {
    int n = 0;

    if (n < max_backends) {
        backends[n++] = SIGLIP_BACKEND_CPU;
    }

    #ifdef GGML_USE_CUDA
    if (n < max_backends) {
        backends[n++] = SIGLIP_BACKEND_CUDA;
    }
    #endif

    #ifdef GGML_USE_METAL
    if (n < max_backends) {
        backends[n++] = SIGLIP_BACKEND_METAL;
    }
    #endif

    #ifdef GGML_USE_VULKAN
    if (n < max_backends) {
        backends[n++] = SIGLIP_BACKEND_VULKAN;
    }
    #endif

    return n;
}

// ============================================================================
// Globale Log-Einstellungen (Deklaration hier, Implementation in core)
// ============================================================================

// Diese Variablen werden in siglip_core.cpp definiert
extern siglip_log_level g_log_level;
extern siglip_log_callback g_log_callback;
extern void * g_log_user_data;

void siglip_set_log_level(siglip_log_level level) {
    g_log_level = level;
}

void siglip_set_log_callback(siglip_log_callback callback, void * user_data) {
    g_log_callback = callback;
    g_log_user_data = user_data;
}
