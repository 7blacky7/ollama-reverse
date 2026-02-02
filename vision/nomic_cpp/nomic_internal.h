/**
 * MODUL: nomic_internal.h
 * ZWECK: Interne Strukturen und Hilfsfunktionen fuer Nomic Vision Encoder
 * INPUT: Keine (Header-Only)
 * OUTPUT: Keine
 * NEBENEFFEKTE: Keine
 * ABHAENGIGKEITEN: nomic.h, ggml (extern)
 * HINWEISE: Nur fuer interne Implementierung - nicht in Public API exponieren
 */

#ifndef NOMIC_INTERNAL_H
#define NOMIC_INTERNAL_H

#include "nomic.h"

#include <string>
#include <vector>

// GGML Headers
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ============================================================================
// Konstanten (intern)
// ============================================================================

// GGUF Magic "GGUF" in Little-Endian
constexpr uint32_t NOMIC_GGUF_MAGIC   = 0x46554747;
constexpr uint32_t NOMIC_GGUF_VERSION = 3;
constexpr size_t   NOMIC_MAX_ERROR_LEN = 512;

// GGUF Metadaten-Typen
enum nomic_gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/** Transformer-Block Tensoren */
struct nomic_layer {
    // Self-Attention
    ggml_tensor * q_weight = nullptr;
    ggml_tensor * q_bias   = nullptr;
    ggml_tensor * k_weight = nullptr;
    ggml_tensor * k_bias   = nullptr;
    ggml_tensor * v_weight = nullptr;
    ggml_tensor * v_bias   = nullptr;
    ggml_tensor * o_weight = nullptr;
    ggml_tensor * o_bias   = nullptr;

    // SwiGLU MLP (gate + up -> down)
    ggml_tensor * ffn_gate_weight = nullptr;
    ggml_tensor * ffn_gate_bias   = nullptr;
    ggml_tensor * ffn_up_weight   = nullptr;
    ggml_tensor * ffn_up_bias     = nullptr;
    ggml_tensor * ffn_down_weight = nullptr;
    ggml_tensor * ffn_down_bias   = nullptr;

    // Layer Normalization
    ggml_tensor * ln1_weight = nullptr;
    ggml_tensor * ln1_bias   = nullptr;
    ggml_tensor * ln2_weight = nullptr;
    ggml_tensor * ln2_bias   = nullptr;
};

/** Vollstaendige Kontext-Struktur */
struct nomic_ctx {
    // Modell-Metadaten
    std::string model_path;
    std::string model_name;
    nomic_hparams hparams;
    int n_threads;

    // GGML Ressourcen
    ggml_context * ctx_data    = nullptr;  // Tensor-Gewichte
    ggml_context * ctx_compute = nullptr;  // Compute-Graph
    ggml_backend_t backend     = nullptr;  // CPU/CUDA Backend
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocr      = nullptr;

    // Vision Encoder Tensoren
    struct {
        // Patch Embedding
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, 3*patch*patch]
        ggml_tensor * patch_embed_bias   = nullptr;  // [hidden]
        ggml_tensor * pos_embed          = nullptr;  // [n_patches+1, hidden]
        ggml_tensor * cls_token          = nullptr;  // [1, hidden]

        // Transformer Blocks
        std::vector<nomic_layer> layers;

        // Output Normalization
        ggml_tensor * post_ln_weight = nullptr;
        ggml_tensor * post_ln_bias   = nullptr;
    } tensors;
};

// ============================================================================
// Globale Variablen (extern - definiert in nomic_core.cpp)
// ============================================================================

extern nomic_log_level g_nomic_log_level;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

void nomic_log_msg(nomic_log_level level, const char * fmt, ...);
void nomic_set_error(const char * fmt, ...);

#define NOMIC_LOG_ERR(...)   nomic_log_msg(NOMIC_LOG_ERROR, __VA_ARGS__)
#define NOMIC_LOG_WARN(...)  nomic_log_msg(NOMIC_LOG_WARN, __VA_ARGS__)
#define NOMIC_LOG_INFO(...)  nomic_log_msg(NOMIC_LOG_INFO, __VA_ARGS__)
#define NOMIC_LOG_DBG(...)   nomic_log_msg(NOMIC_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

std::string nomic_gguf_read_string(FILE * f);
bool nomic_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size);
bool nomic_gguf_skip_value(FILE * f, uint32_t type);

// ============================================================================
// Tensor-Loading und Forward-Pass
// ============================================================================

bool nomic_load_tensors(nomic_ctx * ctx, FILE * f, uint64_t n_tensors);
bool nomic_forward(nomic_ctx * ctx, const float * input, float * output);

#endif // NOMIC_INTERNAL_H
