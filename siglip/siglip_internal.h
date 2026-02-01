/**
 * siglip_internal.h - Interne Strukturen und Hilfsfunktionen
 *
 * Diese Header-Datei enthaelt:
 * - siglip_ctx Struktur (opak in siglip.h)
 * - Interne Hilfsfunktionen
 * - Gemeinsame Konstanten
 */

#ifndef SIGLIP_INTERNAL_H
#define SIGLIP_INTERNAL_H

#include "siglip.h"

#include <string>
#include <vector>

// GGML Headers
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ============================================================================
// Konstanten (intern)
// ============================================================================

// GGUF Magic und Version
constexpr uint32_t SIGLIP_GGUF_MAGIC   = 0x46554747; // "GGUF"
constexpr uint32_t SIGLIP_GGUF_VERSION = 3;

// Maximale String-Laenge fuer Fehler
constexpr size_t SIGLIP_MAX_ERROR_LEN = 512;

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Interner Kontext - haelt alle Modell-Daten und GGML Ressourcen
 */
struct siglip_ctx {
    // Modell-Info
    std::string model_path;
    std::string model_name;
    siglip_hparams hparams;
    siglip_params params;

    // GGML Kontext
    ggml_context * ctx_data = nullptr;      // Tensor-Daten
    ggml_context * ctx_compute = nullptr;   // Compute-Graph
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Tensor-Referenzen
    struct {
        // Patch Embedding
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, channels, patch, patch]
        ggml_tensor * patch_embed_bias = nullptr;    // [hidden]
        ggml_tensor * pos_embed = nullptr;           // [num_patches, hidden]

        // Transformer Blocks
        struct block {
            // Attention
            ggml_tensor * attn_q_weight = nullptr;
            ggml_tensor * attn_q_bias = nullptr;
            ggml_tensor * attn_k_weight = nullptr;
            ggml_tensor * attn_k_bias = nullptr;
            ggml_tensor * attn_v_weight = nullptr;
            ggml_tensor * attn_v_bias = nullptr;
            ggml_tensor * attn_out_weight = nullptr;
            ggml_tensor * attn_out_bias = nullptr;

            // MLP
            ggml_tensor * mlp_fc1_weight = nullptr;
            ggml_tensor * mlp_fc1_bias = nullptr;
            ggml_tensor * mlp_fc2_weight = nullptr;
            ggml_tensor * mlp_fc2_bias = nullptr;

            // LayerNorm
            ggml_tensor * ln1_weight = nullptr;
            ggml_tensor * ln1_bias = nullptr;
            ggml_tensor * ln2_weight = nullptr;
            ggml_tensor * ln2_bias = nullptr;
        };
        std::vector<block> blocks;

        // Output
        ggml_tensor * norm_weight = nullptr;
        ggml_tensor * norm_bias = nullptr;
        ggml_tensor * head_weight = nullptr;  // Optional projection
        ggml_tensor * head_bias = nullptr;
    } tensors;
};

// ============================================================================
// Globale Variablen (extern)
// ============================================================================

// Definiert in siglip_core.cpp
extern siglip_log_level g_log_level;
extern siglip_log_callback g_log_callback;
extern void * g_log_user_data;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

/**
 * Loggt eine Nachricht mit Level-Filter
 */
void siglip_log_msg(siglip_log_level level, const char * fmt, ...);

/**
 * Setzt den letzten Fehler-String (Thread-sicher)
 */
void siglip_set_error(const char * fmt, ...);

// Log-Makros fuer bequeme Nutzung
#define SIGLIP_LOG_ERROR(...) siglip_log_msg(SIGLIP_LOG_ERROR, __VA_ARGS__)
#define SIGLIP_LOG_WARN(...)  siglip_log_msg(SIGLIP_LOG_WARN, __VA_ARGS__)
#define SIGLIP_LOG_INFO(...)  siglip_log_msg(SIGLIP_LOG_INFO, __VA_ARGS__)
#define SIGLIP_LOG_DEBUG(...) siglip_log_msg(SIGLIP_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Funktionen (aus siglip_gguf.cpp)
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string gguf_read_string(FILE * f);

/**
 * Liest einen Metadaten-Wert basierend auf Typ
 */
bool gguf_read_metadata_value(FILE * f, uint32_t type, void * out, size_t max_size);

/**
 * Ueberspringt einen Metadaten-Wert
 */
bool gguf_skip_metadata_value(FILE * f, uint32_t type);

/**
 * Laedt alle Tensoren aus der GGUF-Datei und weist sie dem Kontext zu
 */
bool siglip_load_tensors(siglip_ctx * ctx, FILE * f, uint64_t n_tensors);

#endif // SIGLIP_INTERNAL_H
