/**
 * MODUL: openclip_internal.h
 * ZWECK: Interne Strukturen und Hilfsfunktionen fuer OpenCLIP Vision
 * INPUT: Keine (Header-Only)
 * OUTPUT: Keine
 * NEBENEFFEKTE: Keine
 * ABHAENGIGKEITEN: openclip.h, ggml (extern)
 * HINWEISE: Nur fuer interne Implementierung - nicht in Public API exponieren
 */

#ifndef OPENCLIP_INTERNAL_H
#define OPENCLIP_INTERNAL_H

#include "openclip.h"

#include <string>
#include <vector>
#include <cstdio>

// GGML Headers
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ============================================================================
// Konstanten (intern)
// ============================================================================

// GGUF Magic Bytes "GGUF" in Little-Endian
constexpr uint32_t OPENCLIP_GGUF_MAGIC   = 0x46554747;
constexpr uint32_t OPENCLIP_GGUF_VERSION = 3;

// Maximale Fehler-String-Laenge
constexpr size_t OPENCLIP_MAX_ERROR_LEN = 512;

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Vollstaendige Kontext-Struktur mit allen GGML-Ressourcen
 *
 * OpenCLIP verwendet eine Standard-ViT-Architektur mit:
 * - Pre-Normalization (LayerNorm vor Attention/MLP)
 * - Optional: Attention Pooling statt CLS-Token
 */
struct openclip_ctx {
    // Modell-Metadaten
    std::string model_path;
    std::string model_name;
    openclip_hparams hparams;
    int n_threads;

    // GGML Kontexte und Backend
    ggml_context * ctx_data    = nullptr;   // Tensor-Daten (Gewichte)
    ggml_context * ctx_compute = nullptr;   // Compute-Graph
    ggml_backend_t backend     = nullptr;   // CPU/CUDA Backend
    ggml_backend_buffer_t buffer = nullptr; // Tensor-Buffer
    ggml_gallocr_t allocr      = nullptr;   // Graph-Allocator

    // Tensor-Referenzen fuer Vision Encoder
    struct {
        // Patch Embedding (Conv2D)
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, 3, patch, patch]
        ggml_tensor * patch_embed_bias   = nullptr;  // [hidden]
        ggml_tensor * pos_embed          = nullptr;  // [num_patches + 1, hidden]
        ggml_tensor * cls_token          = nullptr;  // [1, hidden]

        // Pre-LayerNorm (OpenCLIP spezifisch)
        ggml_tensor * pre_ln_weight = nullptr;
        ggml_tensor * pre_ln_bias   = nullptr;

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

            // MLP (Feed-Forward mit QuickGELU)
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

        // Projection Head (optional)
        ggml_tensor * proj_weight = nullptr;
    } tensors;
};

// ============================================================================
// Globale Variablen (extern - definiert in openclip_core.cpp)
// ============================================================================

extern openclip_log_level g_openclip_log_level;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

/**
 * Loggt eine Nachricht mit gegebenem Level
 */
void openclip_log_msg(openclip_log_level level, const char * fmt, ...);

/**
 * Setzt den letzten Fehler-String (Thread-lokal)
 */
void openclip_set_error(const char * fmt, ...);

// Log-Makros fuer einfache Nutzung
#define OPENCLIP_LOG_ERROR(...) openclip_log_msg(OPENCLIP_LOG_ERROR, __VA_ARGS__)
#define OPENCLIP_LOG_WARN(...)  openclip_log_msg(OPENCLIP_LOG_WARN, __VA_ARGS__)
#define OPENCLIP_LOG_INFO(...)  openclip_log_msg(OPENCLIP_LOG_INFO, __VA_ARGS__)
#define OPENCLIP_LOG_DEBUG(...) openclip_log_msg(OPENCLIP_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string openclip_gguf_read_string(FILE * f);

/**
 * Liest Metadaten-Wert basierend auf Typ
 */
bool openclip_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size);

/**
 * Ueberspringt Metadaten-Wert
 */
bool openclip_gguf_skip_value(FILE * f, uint32_t type);

/**
 * Laedt alle Tensoren aus GGUF-Datei
 */
bool openclip_load_tensors(openclip_ctx * ctx, FILE * f, uint64_t n_tensors);

#endif // OPENCLIP_INTERNAL_H
