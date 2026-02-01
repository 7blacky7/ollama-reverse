/**
 * MODUL: dinov2_internal.h
 * ZWECK: Interne Strukturen und Hilfsfunktionen fuer DINOv2 Vision
 * INPUT: Keine (Header-Only)
 * OUTPUT: Keine
 * NEBENEFFEKTE: Keine
 * ABHAENGIGKEITEN: dinov2.h, ggml (extern)
 * HINWEISE: Nur fuer interne Implementierung - nicht in Public API exponieren
 *
 * DINOv2 Besonderheiten:
 * - Self-supervised Learning (kein Text-Encoder)
 * - Registrierte Tokens (optional, meist 4 extra Tokens)
 * - Keine CLIP-Style Projektion, direkte Features
 */

#ifndef DINOV2_INTERNAL_H
#define DINOV2_INTERNAL_H

#include "dinov2.h"

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
constexpr uint32_t DINOV2_GGUF_MAGIC   = 0x46554747;
constexpr uint32_t DINOV2_GGUF_VERSION = 3;

// Maximale Fehler-String-Laenge
constexpr size_t DINOV2_MAX_ERROR_LEN = 512;

// Default-Werte fuer DINOv2 Varianten
constexpr int DINOV2_S_HIDDEN  = 384;   // Small
constexpr int DINOV2_B_HIDDEN  = 768;   // Base
constexpr int DINOV2_L_HIDDEN  = 1024;  // Large
constexpr int DINOV2_G_HIDDEN  = 1536;  // Giant

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Vollstaendige Kontext-Struktur mit allen GGML-Ressourcen
 */
struct dinov2_ctx {
    // Modell-Metadaten
    std::string model_path;
    std::string model_name;
    dinov2_hparams hparams;
    int n_threads;

    // DINOv2-spezifisch: Anzahl Register-Tokens (oft 0 oder 4)
    int num_register_tokens;

    // GGML Kontexte und Backend
    ggml_context * ctx_data    = nullptr;   // Tensor-Daten (Gewichte)
    ggml_context * ctx_compute = nullptr;   // Compute-Graph
    ggml_backend_t backend     = nullptr;   // CPU/CUDA Backend
    ggml_backend_buffer_t buffer = nullptr; // Tensor-Buffer
    ggml_gallocr_t allocr      = nullptr;   // Graph-Allocator

    // Tensor-Referenzen fuer Vision Encoder
    struct {
        // Patch Embedding (Linear projection of flattened patches)
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, 3 * patch * patch]
        ggml_tensor * patch_embed_bias   = nullptr;  // [hidden]
        ggml_tensor * pos_embed          = nullptr;  // [num_patches + 1 + reg, hidden]
        ggml_tensor * cls_token          = nullptr;  // [1, hidden]
        ggml_tensor * register_tokens    = nullptr;  // [num_reg, hidden] (optional)

        // Transformer Bloecke
        struct layer {
            // Self-Attention (QKV als kombinierter Tensor oder separat)
            ggml_tensor * q_weight = nullptr;
            ggml_tensor * q_bias   = nullptr;
            ggml_tensor * k_weight = nullptr;
            ggml_tensor * k_bias   = nullptr;
            ggml_tensor * v_weight = nullptr;
            ggml_tensor * v_bias   = nullptr;
            ggml_tensor * o_weight = nullptr;
            ggml_tensor * o_bias   = nullptr;

            // MLP (Feed-Forward mit GELU)
            ggml_tensor * ff_up_weight   = nullptr;
            ggml_tensor * ff_up_bias     = nullptr;
            ggml_tensor * ff_down_weight = nullptr;
            ggml_tensor * ff_down_bias   = nullptr;

            // Layer Normalization (Pre-LN Architektur)
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
// Globale Variablen (extern - definiert in dinov2_core.cpp)
// ============================================================================

extern dinov2_log_level g_dinov2_log_level;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

/**
 * Loggt eine Nachricht mit gegebenem Level
 */
void dinov2_log_msg(dinov2_log_level level, const char * fmt, ...);

/**
 * Setzt den letzten Fehler-String (Thread-lokal)
 */
void dinov2_set_error(const char * fmt, ...);

// Log-Makros fuer einfache Nutzung
#define DINOV2_LOG_ERROR(...) dinov2_log_msg(DINOV2_LOG_ERROR, __VA_ARGS__)
#define DINOV2_LOG_WARN(...)  dinov2_log_msg(DINOV2_LOG_WARN, __VA_ARGS__)
#define DINOV2_LOG_INFO(...)  dinov2_log_msg(DINOV2_LOG_INFO, __VA_ARGS__)
#define DINOV2_LOG_DEBUG(...) dinov2_log_msg(DINOV2_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string dinov2_gguf_read_string(FILE * f);

/**
 * Liest Metadaten-Wert basierend auf Typ
 */
bool dinov2_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size);

/**
 * Ueberspringt Metadaten-Wert
 */
bool dinov2_gguf_skip_value(FILE * f, uint32_t type);

/**
 * Laedt alle Tensoren aus GGUF-Datei
 */
bool dinov2_load_tensors(dinov2_ctx * ctx, FILE * f, uint64_t n_tensors);

#endif // DINOV2_INTERNAL_H
