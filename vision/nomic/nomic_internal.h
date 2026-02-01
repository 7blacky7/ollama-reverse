/**
 * MODUL: nomic_internal.h
 * ZWECK: Interne Strukturen und Hilfsfunktionen fuer Nomic Vision
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

// GGUF Magic Bytes "GGUF" in Little-Endian
constexpr uint32_t NOMIC_GGUF_MAGIC   = 0x46554747;
constexpr uint32_t NOMIC_GGUF_VERSION = 3;

// Maximale Fehler-String-Laenge
constexpr size_t NOMIC_MAX_ERROR_LEN = 512;

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Vollstaendige Kontext-Struktur mit allen GGML-Ressourcen
 */
struct nomic_ctx {
    // Modell-Metadaten
    std::string model_path;
    std::string model_name;
    nomic_hparams hparams;
    int n_threads;

    // GGML Kontexte und Backend
    ggml_context * ctx_data    = nullptr;   // Tensor-Daten (Gewichte)
    ggml_context * ctx_compute = nullptr;   // Compute-Graph
    ggml_backend_t backend     = nullptr;   // CPU/CUDA Backend
    ggml_backend_buffer_t buffer = nullptr; // Tensor-Buffer
    ggml_gallocr_t allocr      = nullptr;   // Graph-Allocator

    // Tensor-Referenzen fuer Vision Encoder
    struct {
        // Patch Embedding (Conv2D als Dense)
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, 3, patch, patch]
        ggml_tensor * patch_embed_bias   = nullptr;  // [hidden]
        ggml_tensor * pos_embed          = nullptr;  // [num_patches + 1, hidden]
        ggml_tensor * cls_token          = nullptr;  // [1, hidden]

        // Transformer Bloecke
        struct layer {
            // Self-Attention
            ggml_tensor * q_weight = nullptr;
            ggml_tensor * q_bias   = nullptr;
            ggml_tensor * k_weight = nullptr;
            ggml_tensor * k_bias   = nullptr;
            ggml_tensor * v_weight = nullptr;
            ggml_tensor * v_bias   = nullptr;
            ggml_tensor * o_weight = nullptr;
            ggml_tensor * o_bias   = nullptr;

            // MLP (Feed-Forward)
            ggml_tensor * ff_up_weight   = nullptr;
            ggml_tensor * ff_up_bias     = nullptr;
            ggml_tensor * ff_down_weight = nullptr;
            ggml_tensor * ff_down_bias   = nullptr;

            // Layer Normalization
            ggml_tensor * ln1_weight = nullptr;
            ggml_tensor * ln1_bias   = nullptr;
            ggml_tensor * ln2_weight = nullptr;
            ggml_tensor * ln2_bias   = nullptr;
        };
        std::vector<layer> layers;

        // Output Normalization
        ggml_tensor * final_ln_weight = nullptr;
        ggml_tensor * final_ln_bias   = nullptr;
    } tensors;
};

// ============================================================================
// Globale Variablen (extern - definiert in nomic_core.cpp)
// ============================================================================

extern nomic_log_level g_nomic_log_level;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

/**
 * Loggt eine Nachricht mit gegebenem Level
 */
void nomic_log_msg(nomic_log_level level, const char * fmt, ...);

/**
 * Setzt den letzten Fehler-String (Thread-lokal)
 */
void nomic_set_error(const char * fmt, ...);

// Log-Makros fuer einfache Nutzung
#define NOMIC_LOG_ERROR(...) nomic_log_msg(NOMIC_LOG_ERROR, __VA_ARGS__)
#define NOMIC_LOG_WARN(...)  nomic_log_msg(NOMIC_LOG_WARN, __VA_ARGS__)
#define NOMIC_LOG_INFO(...)  nomic_log_msg(NOMIC_LOG_INFO, __VA_ARGS__)
#define NOMIC_LOG_DEBUG(...) nomic_log_msg(NOMIC_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string nomic_gguf_read_string(FILE * f);

/**
 * Liest Metadaten-Wert basierend auf Typ
 */
bool nomic_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size);

/**
 * Ueberspringt Metadaten-Wert
 */
bool nomic_gguf_skip_value(FILE * f, uint32_t type);

/**
 * Laedt alle Tensoren aus GGUF-Datei
 */
bool nomic_load_tensors(nomic_ctx * ctx, FILE * f, uint64_t n_tensors);

#endif // NOMIC_INTERNAL_H
