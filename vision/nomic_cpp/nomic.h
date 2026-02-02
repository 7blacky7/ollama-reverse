/**
 * MODUL: nomic.h
 * ZWECK: Nomic Embed Vision C API - GGUF basierter Vision Encoder
 * INPUT: GGUF Modellpfad, RGB Bilddaten
 * OUTPUT: Embedding-Vektoren (768-dim float, L2-normalisiert)
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O beim Laden
 * ABHAENGIGKEITEN: ggml, ggml-alloc, ggml-backend (extern)
 * HINWEISE: API-Kompatibel mit SigLIP-Struktur, GGUF v3 Format
 *
 * Architektur nomic-embed-vision-v1.5:
 * - n_embd: 768, n_head: 12, n_layer: 12
 * - img_size: 384, patch_size: 14 (729 Patches)
 * - Aktivierung: SwiGLU
 * - Pooling: CLS-Token (Index 0)
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
// Typen
// ============================================================================

/** Log-Level fuer Diagnose */
enum nomic_log_level {
    NOMIC_LOG_NONE  = 0,
    NOMIC_LOG_ERROR = 1,
    NOMIC_LOG_WARN  = 2,
    NOMIC_LOG_INFO  = 3,
    NOMIC_LOG_DEBUG = 4
};

/** Modell-Hyperparameter (readonly nach Init) */
struct nomic_hparams {
    int hidden_size;            // 768
    int intermediate_size;      // 3072 (4 * hidden)
    int num_attention_heads;    // 12
    int num_hidden_layers;      // 12
    int image_size;             // 384
    int patch_size;             // 14
    int num_patches;            // 729 (27 * 27)
    float layer_norm_eps;       // 1e-6
};

/** Opaker Kontext-Typ */
typedef struct nomic_ctx nomic_ctx;

/** Embedding-Resultat */
struct nomic_embedding {
    float * data;       // Embedding-Daten
    int     dim;        // Dimension (768)
    int     batch_size; // Anzahl Embeddings
    bool    normalized; // L2-normalisiert?
};

// ============================================================================
// Lifecycle
// ============================================================================

/**
 * Laedt Nomic Vision Modell aus GGUF-Datei
 * @param model_path  Pfad zur .gguf Datei
 * @param n_threads   CPU-Threads (0 = auto)
 * @return Kontext oder NULL bei Fehler
 */
nomic_ctx * nomic_init(const char * model_path, int n_threads);

/** Gibt alle Ressourcen frei */
void nomic_free(nomic_ctx * ctx);

// ============================================================================
// Encoding
// ============================================================================

/**
 * Encodiert ein Bild (RGB, HWC) zu Embedding
 * @param ctx         Nomic Kontext
 * @param image_data  RGB Pixeldaten (HWC, uint8)
 * @param width       Bildbreite
 * @param height      Bildhoehe
 * @return Embedding oder NULL bei Fehler
 */
struct nomic_embedding * nomic_encode_image(
    nomic_ctx * ctx,
    const uint8_t * image_data,
    int width, int height
);

/**
 * Encodiert bereits vorverarbeitete Daten (CHW, float, normalisiert)
 * @param ctx           Nomic Kontext
 * @param preprocessed  Float-Array [3, image_size, image_size]
 * @return Embedding oder NULL bei Fehler
 */
struct nomic_embedding * nomic_encode_preprocessed(
    nomic_ctx * ctx,
    const float * preprocessed
);

/** Gibt Embedding-Speicher frei */
void nomic_embedding_free(struct nomic_embedding * emb);

// ============================================================================
// Info-Abfragen
// ============================================================================

/** Embedding-Dimension (768) */
int nomic_get_embedding_dim(const nomic_ctx * ctx);

/** Erwartete Bildgroesse (384) */
int nomic_get_image_size(const nomic_ctx * ctx);

/** Patch-Groesse (14) */
int nomic_get_patch_size(const nomic_ctx * ctx);

/** Hyperparameter-Struct */
const struct nomic_hparams * nomic_get_hparams(const nomic_ctx * ctx);

// ============================================================================
// Utilities
// ============================================================================

/** L2-Normalisierung in-place */
void nomic_normalize(float * data, int size);

/** Cosine-Similarity */
float nomic_cosine_similarity(const float * a, const float * b, int size);

// ============================================================================
// Fehler und Logging
// ============================================================================

/** Letzter Fehler (oder NULL) */
const char * nomic_get_last_error(void);

/** Fehler loeschen */
void nomic_clear_error(void);

/** Log-Level setzen */
void nomic_set_log_level(enum nomic_log_level level);

#ifdef __cplusplus
}
#endif

#endif // NOMIC_H
