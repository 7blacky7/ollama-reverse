/**
 * MODUL: dinov2.h
 * ZWECK: DINOv2 Vision C API - Self-Supervised Vision Features (NUR Bild, kein Text)
 * INPUT: Modellpfad, Bilddaten (RGB), Thread-Anzahl, Output-Modus
 * OUTPUT: Vision Feature-Vektoren (CLS/Patches/Mean)
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O beim Laden
 * ABHAENGIGKEITEN: ggml (extern), ggml-alloc (extern), ggml-backend (extern)
 * HINWEISE: DINOv2 ist reines Vision-Modell ohne Text-Encoder
 *           Bietet CLS Token, Patch Tokens oder Mean Pooling als Output
 */

#ifndef DINOV2_H
#define DINOV2_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Konstanten
// ============================================================================

#define DINOV2_DEFAULT_IMAGE_SIZE   518
#define DINOV2_DEFAULT_PATCH_SIZE   14
#define DINOV2_DEFAULT_HIDDEN_SIZE  768
#define DINOV2_MAX_LAYERS           12
#define DINOV2_MAX_HEADS            12

// ============================================================================
// Enumerationen
// ============================================================================

/**
 * Log-Level fuer Diagnose-Ausgaben
 */
enum dinov2_log_level {
    DINOV2_LOG_NONE  = 0,
    DINOV2_LOG_ERROR = 1,
    DINOV2_LOG_WARN  = 2,
    DINOV2_LOG_INFO  = 3,
    DINOV2_LOG_DEBUG = 4
};

/**
 * Output-Modus: Bestimmt welche Features zurueckgegeben werden
 *
 * DINOV2_OUTPUT_CLS:     Nur CLS Token (1 x dim) - Standard fuer Klassifikation
 * DINOV2_OUTPUT_PATCHES: Alle Patch Tokens (N x dim) - fuer Dense Prediction
 * DINOV2_OUTPUT_MEAN:    Mean ueber alle Patches (1 x dim) - Alternative zu CLS
 */
typedef enum {
    DINOV2_OUTPUT_CLS     = 0,
    DINOV2_OUTPUT_PATCHES = 1,
    DINOV2_OUTPUT_MEAN    = 2
} dinov2_output_mode;

// ============================================================================
// Datenstrukturen
// ============================================================================

/**
 * Modell-Hyperparameter (schreibgeschuetzt nach Laden)
 */
struct dinov2_hparams {
    int hidden_size;            // Embedding-Dimension (384/768/1024/1536)
    int intermediate_size;      // MLP Hidden Size
    int num_attention_heads;    // Attention Heads
    int num_hidden_layers;      // Transformer Layers
    int image_size;             // Input-Bildgroesse (518 fuer ViT-L)
    int patch_size;             // Patch-Groesse (14)
    int num_patches;            // Anzahl Patches ohne CLS (1369 = 37*37)
    float layer_norm_eps;       // LayerNorm Epsilon (1e-6)
};

/**
 * Kontext (opak - Details in dinov2_internal.h)
 */
typedef struct dinov2_ctx dinov2_ctx;

// ============================================================================
// Initialisierung und Freigabe
// ============================================================================

/**
 * Laedt DINOv2 Modell aus GGUF-Datei
 *
 * @param model_path  Pfad zur GGUF-Datei
 * @param n_threads   Anzahl CPU-Threads (0 = auto)
 * @return            Kontext oder NULL bei Fehler
 */
dinov2_ctx * dinov2_load(const char * model_path, int n_threads);

/**
 * Gibt alle Ressourcen des Kontexts frei
 */
void dinov2_free(dinov2_ctx * ctx);

// ============================================================================
// Encoding
// ============================================================================

/**
 * Encodiert ein Bild zu Vision Features
 *
 * @param ctx         DINOv2 Kontext
 * @param image_data  RGB Pixeldaten (HWC Format, 8-bit pro Kanal)
 * @param image_len   Groesse der Bilddaten in Bytes
 * @param embedding   Output-Array fuer Features (Groesse abhaengig von mode)
 * @param max_dim     Maximale Dimension des Output-Arrays
 * @param mode        Output-Modus (CLS/PATCHES/MEAN)
 * @return            Tatsaechliche Anzahl floats oder -1 bei Fehler
 *                    - CLS/MEAN: hidden_size
 *                    - PATCHES: num_patches * hidden_size
 */
int dinov2_encode(
    dinov2_ctx * ctx,
    const uint8_t * image_data,
    size_t image_len,
    float * embedding,
    int max_dim,
    dinov2_output_mode mode
);

// ============================================================================
// Modell-Info
// ============================================================================

/**
 * Gibt die Embedding-Dimension zurueck (hidden_size)
 */
int dinov2_get_dim(const dinov2_ctx * ctx);

/**
 * Gibt die Anzahl der Patches zurueck (ohne CLS Token)
 */
int dinov2_get_num_patches(const dinov2_ctx * ctx);

/**
 * Gibt die erwartete Bildgroesse zurueck
 */
int dinov2_get_image_size(const dinov2_ctx * ctx);

/**
 * Gibt die Hyperparameter zurueck
 */
const struct dinov2_hparams * dinov2_get_hparams(const dinov2_ctx * ctx);

// ============================================================================
// Fehlerbehandlung und Logging
// ============================================================================

/**
 * Gibt den letzten Fehler zurueck (oder NULL)
 */
const char * dinov2_get_last_error(void);

/**
 * Loescht den letzten Fehler
 */
void dinov2_clear_error(void);

/**
 * Setzt das Log-Level
 */
void dinov2_set_log_level(enum dinov2_log_level level);

#ifdef __cplusplus
}
#endif

#endif // DINOV2_H
