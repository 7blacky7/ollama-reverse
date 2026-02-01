/**
 * MODUL: evaclip_internal.h
 * ZWECK: Interne Strukturen und Hilfsfunktionen fuer EVA-CLIP Vision
 * INPUT: Keine (Header-Only)
 * OUTPUT: Keine
 * NEBENEFFEKTE: Keine
 * ABHAENGIGKEITEN: evaclip.h, ggml (extern)
 * HINWEISE: Nur fuer interne Implementierung - nicht in Public API exponieren
 *
 * EVA-CLIP verwendet:
 * - EVA-Initialisierung (aus Masked Autoencoder Training)
 * - Rotary Position Embeddings (optional)
 * - SwiGLU MLP Aktivierung
 */

#ifndef EVACLIP_INTERNAL_H
#define EVACLIP_INTERNAL_H

#include "evaclip.h"

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
constexpr uint32_t EVACLIP_GGUF_MAGIC   = 0x46554747;
constexpr uint32_t EVACLIP_GGUF_VERSION = 3;

// Maximale Fehler-String-Laenge
constexpr size_t EVACLIP_MAX_ERROR_LEN = 512;

// Default-Werte fuer EVA02-CLIP-L-14
constexpr int EVACLIP_DEFAULT_HIDDEN_SIZE   = 1024;
constexpr int EVACLIP_DEFAULT_IMAGE_SIZE    = 336;
constexpr int EVACLIP_DEFAULT_PATCH_SIZE    = 14;
constexpr int EVACLIP_DEFAULT_LAYERS        = 24;
constexpr int EVACLIP_DEFAULT_HEADS         = 16;
constexpr int EVACLIP_DEFAULT_INTERMEDIATE  = 4096;

// ============================================================================
// Log-Level Definition (intern)
// ============================================================================

enum evaclip_log_level {
    EVACLIP_LOG_NONE  = 0,
    EVACLIP_LOG_ERROR = 1,
    EVACLIP_LOG_WARN  = 2,
    EVACLIP_LOG_INFO  = 3,
    EVACLIP_LOG_DEBUG = 4
};

// ============================================================================
// Hyperparameter-Struktur (intern)
// ============================================================================

struct evaclip_hparams {
    int hidden_size;            // Embedding-Dimension (768, 1024)
    int intermediate_size;      // MLP Hidden Size
    int num_attention_heads;    // Attention Heads
    int num_hidden_layers;      // Transformer Layers
    int image_size;             // Input-Bildgroesse (224, 336)
    int patch_size;             // Patch-Groesse (14, 16)
    int num_patches;            // Anzahl Patches
    float layer_norm_eps;       // LayerNorm Epsilon (1e-6)

    // Preprocessing-Parameter (ImageNet-Standard fuer CLIP)
    float image_mean[3];        // {0.48145466, 0.4578275, 0.40821073}
    float image_std[3];         // {0.26862954, 0.26130258, 0.27577711}
};

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Vollstaendige Kontext-Struktur mit allen GGML-Ressourcen
 */
struct evaclip_ctx {
    // Modell-Metadaten
    std::string model_path;
    std::string model_name;
    evaclip_hparams hparams;
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
            // Self-Attention (Q, K, V, Output)
            ggml_tensor * q_weight = nullptr;
            ggml_tensor * q_bias   = nullptr;
            ggml_tensor * k_weight = nullptr;
            ggml_tensor * k_bias   = nullptr;
            ggml_tensor * v_weight = nullptr;
            ggml_tensor * v_bias   = nullptr;
            ggml_tensor * o_weight = nullptr;
            ggml_tensor * o_bias   = nullptr;

            // MLP (EVA nutzt oft SwiGLU-aehnliche Struktur)
            ggml_tensor * ff_up_weight   = nullptr;
            ggml_tensor * ff_up_bias     = nullptr;
            ggml_tensor * ff_down_weight = nullptr;
            ggml_tensor * ff_down_bias   = nullptr;

            // Layer Normalization (Pre-LN in EVA)
            ggml_tensor * ln1_weight = nullptr;
            ggml_tensor * ln1_bias   = nullptr;
            ggml_tensor * ln2_weight = nullptr;
            ggml_tensor * ln2_bias   = nullptr;
        };
        std::vector<layer> layers;

        // Output Normalization
        ggml_tensor * final_ln_weight = nullptr;
        ggml_tensor * final_ln_bias   = nullptr;

        // Optional: Projection Head
        ggml_tensor * head_weight = nullptr;
        ggml_tensor * head_bias   = nullptr;
    } tensors;
};

// ============================================================================
// Globale Variablen (extern - definiert in evaclip_core.cpp)
// ============================================================================

extern evaclip_log_level g_evaclip_log_level;

// ============================================================================
// Interne Logging-Funktionen
// ============================================================================

/**
 * Loggt eine Nachricht mit gegebenem Level
 */
void evaclip_log_msg(evaclip_log_level level, const char * fmt, ...);

/**
 * Setzt den letzten Fehler-String (Thread-lokal)
 */
void evaclip_set_error(const char * fmt, ...);

// Log-Makros fuer einfache Nutzung
#define EVACLIP_LOG_ERROR(...) evaclip_log_msg(EVACLIP_LOG_ERROR, __VA_ARGS__)
#define EVACLIP_LOG_WARN(...)  evaclip_log_msg(EVACLIP_LOG_WARN, __VA_ARGS__)
#define EVACLIP_LOG_INFO(...)  evaclip_log_msg(EVACLIP_LOG_INFO, __VA_ARGS__)
#define EVACLIP_LOG_DEBUG(...) evaclip_log_msg(EVACLIP_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string evaclip_gguf_read_string(FILE * f);

/**
 * Liest Metadaten-Wert basierend auf Typ
 */
bool evaclip_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size);

/**
 * Ueberspringt Metadaten-Wert
 */
bool evaclip_gguf_skip_value(FILE * f, uint32_t type);

/**
 * Laedt alle Tensoren aus GGUF-Datei
 */
bool evaclip_load_tensors(evaclip_ctx * ctx, FILE * f, uint64_t n_tensors);

#endif // EVACLIP_INTERNAL_H
