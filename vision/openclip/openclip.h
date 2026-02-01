/**
 * MODUL: openclip.h
 * ZWECK: OpenCLIP Vision Encoder C API - Groessere CLIP-Variante von LAION
 * INPUT: Modell-Pfad, Bild-Daten, Parameter
 * OUTPUT: Float32 Embeddings (bis 1280-dim fuer ViT-bigG-14)
 * NEBENEFFEKTE: Laedt Modell, alloziert Speicher, Datei-I/O
 * ABHAENGIGKEITEN: ggml (fuer GGUF-Format)
 * HINWEISE: OpenCLIP unterstuetzt groessere Modelle als Standard-CLIP
 *           ViT-bigG-14: 1.8B Parameter, 1280-dim Embeddings, 40 Layers
 *           Trainiert auf LAION-2B Dataset fuer bessere Zero-Shot Performance
 */

#ifndef OPENCLIP_H
#define OPENCLIP_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Konstanten
// ============================================================================

// ViT-bigG-14 Defaults (groesstes OpenCLIP Modell)
#define OPENCLIP_DEFAULT_IMAGE_SIZE   224
#define OPENCLIP_DEFAULT_PATCH_SIZE   14
#define OPENCLIP_DEFAULT_HIDDEN_SIZE  1280
#define OPENCLIP_MAX_LAYERS           40
#define OPENCLIP_MAX_HEADS            20

// Fehler-Codes
#define OPENCLIP_SUCCESS        0
#define OPENCLIP_ERR_NULL_CTX  -1
#define OPENCLIP_ERR_NULL_IMG  -2
#define OPENCLIP_ERR_DECODE    -3
#define OPENCLIP_ERR_ENCODE    -4
#define OPENCLIP_ERR_ALLOC     -5

// ============================================================================
// Datenstrukturen
// ============================================================================

/**
 * Log-Level fuer Diagnose-Ausgaben
 */
enum openclip_log_level {
    OPENCLIP_LOG_NONE  = 0,
    OPENCLIP_LOG_ERROR = 1,
    OPENCLIP_LOG_WARN  = 2,
    OPENCLIP_LOG_INFO  = 3,
    OPENCLIP_LOG_DEBUG = 4
};

/**
 * Modell-Hyperparameter (schreibgeschuetzt nach Laden)
 */
struct openclip_hparams {
    int hidden_size;            // Embedding-Dimension (768, 1024, 1280)
    int intermediate_size;      // MLP Hidden Size
    int num_attention_heads;    // Attention Heads (12, 16, 20)
    int num_hidden_layers;      // Transformer Layers (12, 24, 40)
    int image_size;             // Input-Bildgroesse (224, 336)
    int patch_size;             // Patch-Groesse (14, 16)
    int num_patches;            // Anzahl Patches
    float layer_norm_eps;       // LayerNorm Epsilon (1e-5 fuer OpenCLIP)
};

/**
 * Kontext (opak)
 */
typedef struct openclip_ctx openclip_ctx;

/**
 * Initialisierungsparameter (fuer CGO Kompatibilitaet)
 */
typedef struct {
    int32_t n_threads;      // Anzahl CPU-Threads
    int32_t n_gpu_layers;   // Anzahl GPU-Layers (-1 = alle)
    int32_t main_gpu;       // Index des Haupt-GPUs
    int8_t  use_mmap;       // Memory-Mapping aktivieren
    int8_t  use_mlock;      // Memory-Locking aktivieren
} openclip_init_params;

/**
 * Modell-Information (fuer CGO Kompatibilitaet)
 */
typedef struct {
    const char* name;        // Modell-Name (z.B. "ViT-bigG-14")
    int32_t embedding_dim;   // Embedding-Dimension
    int32_t image_size;      // Erwartete Bildgroesse
} openclip_model_info;

// ============================================================================
// Initialisierung und Freigabe
// ============================================================================

/**
 * Standard-Parameter zurueckgeben
 */
openclip_init_params openclip_default_params(void);

/**
 * Laedt OpenCLIP Vision Modell aus GGUF-Datei (neue API)
 *
 * @param model_path  Pfad zur GGUF-Datei
 * @param n_threads   Anzahl CPU-Threads (0 = auto)
 * @return            Kontext oder NULL bei Fehler
 */
openclip_ctx * openclip_load(const char * model_path, int n_threads);

/**
 * Laedt OpenCLIP Modell mit erweiterten Parametern (CGO Kompatibilitaet)
 */
openclip_ctx* openclip_init(const char* model_path, openclip_init_params params);

/**
 * Gibt alle Ressourcen des Kontexts frei
 */
void openclip_free(openclip_ctx * ctx);

// ============================================================================
// Encoding
// ============================================================================

/**
 * Encodiert ein Bild zu einem Embedding-Vektor (neue API)
 *
 * @param ctx         OpenCLIP Kontext
 * @param image_data  RGB Pixeldaten (HWC Format, 8-bit pro Kanal)
 * @param image_len   Groesse der Bilddaten in Bytes
 * @param embedding   Output-Array fuer Embedding (min. max_dim floats)
 * @param max_dim     Maximale Dimension des Output-Arrays
 * @return            Tatsaechliche Embedding-Dimension oder -1 bei Fehler
 */
int openclip_encode(
    openclip_ctx * ctx,
    const uint8_t * image_data,
    size_t image_len,
    float * embedding,
    int max_dim
);

/**
 * Encodiert ein einzelnes Bild (CGO Kompatibilitaet)
 */
int32_t openclip_encode_image(
    openclip_ctx* ctx,
    const uint8_t* image_data,
    size_t image_size,
    float* embedding_out,
    int32_t embedding_dim
);

/**
 * Encodiert einen Batch von Bildern (CGO Kompatibilitaet)
 */
int32_t openclip_encode_batch(
    openclip_ctx* ctx,
    const uint8_t** images,
    const size_t* sizes,
    int32_t batch_size,
    float* embeddings_out,
    int32_t embedding_dim
);

// ============================================================================
// Modell-Info
// ============================================================================

/**
 * Gibt die Embedding-Dimension zurueck
 */
int openclip_get_dim(const openclip_ctx * ctx);

/**
 * Gibt die erwartete Bildgroesse zurueck
 */
int openclip_get_image_size(const openclip_ctx * ctx);

/**
 * Gibt die Hyperparameter zurueck
 */
const struct openclip_hparams * openclip_get_hparams(const openclip_ctx * ctx);

/**
 * Gibt Modell-Information zurueck (CGO Kompatibilitaet)
 */
openclip_model_info openclip_get_model_info(openclip_ctx* ctx);

// ============================================================================
// Fehlerbehandlung und Logging
// ============================================================================

/**
 * Gibt den letzten Fehler zurueck (oder NULL)
 */
const char * openclip_get_last_error(void);

/**
 * Loescht den letzten Fehler
 */
void openclip_clear_error(void);

/**
 * Setzt das Log-Level
 */
void openclip_set_log_level(enum openclip_log_level level);

#ifdef __cplusplus
}
#endif

#endif // OPENCLIP_H
