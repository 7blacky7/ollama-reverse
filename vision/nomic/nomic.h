/**
 * MODUL: nomic.h
 * ZWECK: Nomic Embed Vision C API - Unified Text+Image Embedding Space
 * INPUT: Modellpfad, Bilddaten (RGB), Thread-Anzahl
 * OUTPUT: Embedding-Vektoren (768-dim float)
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O beim Laden
 * ABHAENGIGKEITEN: ggml (extern), ggml-alloc (extern), ggml-backend (extern)
 * HINWEISE: API kompatibel mit SigLIP-Struktur, GGUF v2/v3 unterstuetzt
 */

#ifndef NOMIC_H
#define NOMIC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Konstanten
// ============================================================================

#define NOMIC_DEFAULT_IMAGE_SIZE   384
#define NOMIC_DEFAULT_PATCH_SIZE   14
#define NOMIC_DEFAULT_HIDDEN_SIZE  768
#define NOMIC_MAX_LAYERS           12
#define NOMIC_MAX_HEADS            12

// ============================================================================
// Datenstrukturen
// ============================================================================

/**
 * Log-Level fuer Diagnose-Ausgaben
 */
enum nomic_log_level {
    NOMIC_LOG_NONE  = 0,
    NOMIC_LOG_ERROR = 1,
    NOMIC_LOG_WARN  = 2,
    NOMIC_LOG_INFO  = 3,
    NOMIC_LOG_DEBUG = 4
};

/**
 * Modell-Hyperparameter (schreibgeschuetzt nach Laden)
 */
struct nomic_hparams {
    int hidden_size;            // Embedding-Dimension (768)
    int intermediate_size;      // MLP Hidden Size (3072)
    int num_attention_heads;    // Attention Heads (12)
    int num_hidden_layers;      // Transformer Layers (12)
    int image_size;             // Input-Bildgroesse (384)
    int patch_size;             // Patch-Groesse (14)
    int num_patches;            // Anzahl Patches (729 = 27*27)
    float layer_norm_eps;       // LayerNorm Epsilon (1e-6)
};

/**
 * Kontext (opak - Details in nomic_internal.h)
 */
typedef struct nomic_ctx nomic_ctx;

// ============================================================================
// Initialisierung und Freigabe
// ============================================================================

/**
 * Laedt Nomic Vision Modell aus GGUF-Datei
 *
 * @param model_path  Pfad zur GGUF-Datei
 * @param n_threads   Anzahl CPU-Threads (0 = auto)
 * @return            Kontext oder NULL bei Fehler
 */
nomic_ctx * nomic_load_model(const char * model_path, int n_threads);

/**
 * Gibt alle Ressourcen des Kontexts frei
 */
void nomic_free(nomic_ctx * ctx);

// ============================================================================
// Encoding
// ============================================================================

/**
 * Encodiert ein Bild zu einem Embedding-Vektor
 *
 * @param ctx         Nomic Kontext
 * @param image_data  RGB Pixeldaten (HWC Format, 8-bit pro Kanal)
 * @param image_len   Groesse der Bilddaten in Bytes
 * @param embedding   Output-Array fuer Embedding (min. max_dim floats)
 * @param max_dim     Maximale Dimension des Output-Arrays
 * @return            Tatsaechliche Embedding-Dimension oder -1 bei Fehler
 */
int nomic_encode(
    nomic_ctx * ctx,
    const uint8_t * image_data,
    size_t image_len,
    float * embedding,
    int max_dim
);

// ============================================================================
// Modell-Info
// ============================================================================

/**
 * Gibt die Embedding-Dimension zurueck
 */
int nomic_get_embedding_dim(const nomic_ctx * ctx);

/**
 * Gibt die erwartete Bildgroesse zurueck
 */
int nomic_get_image_size(const nomic_ctx * ctx);

/**
 * Gibt die Hyperparameter zurueck
 */
const struct nomic_hparams * nomic_get_hparams(const nomic_ctx * ctx);

// ============================================================================
// Fehlerbehandlung und Logging
// ============================================================================

/**
 * Gibt den letzten Fehler zurueck (oder NULL)
 */
const char * nomic_get_last_error(void);

/**
 * Loescht den letzten Fehler
 */
void nomic_clear_error(void);

/**
 * Setzt das Log-Level
 */
void nomic_set_log_level(enum nomic_log_level level);

#ifdef __cplusplus
}
#endif

#endif // NOMIC_H
