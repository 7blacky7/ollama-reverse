/**
 * siglip.h - SigLIP Vision Encoder für llama.cpp
 *
 * Standalone Image-Embedding-Generierung mit SigLIP (Sigmoid Loss for Language Image Pre-Training).
 * Unterstützt ViT-B/16, ViT-L/16 und ViT-SO400M Modelle.
 *
 * Verwendung:
 *   siglip_ctx * ctx = siglip_load_model("siglip-vit-b.gguf", params);
 *   siglip_image img = siglip_image_load("image.jpg");
 *   float * embedding = siglip_encode(ctx, &img);
 *   siglip_free(ctx);
 */

#ifndef SIGLIP_H
#define SIGLIP_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Konstanten
// ============================================================================

#define SIGLIP_MAX_IMAGE_SIZE    384
#define SIGLIP_MAX_PATCH_SIZE    16
#define SIGLIP_MAX_HIDDEN_SIZE   1024
#define SIGLIP_MAX_LAYERS        24
#define SIGLIP_MAX_HEADS         16

// ============================================================================
// Datenstrukturen
// ============================================================================

/**
 * Modell-Typ
 */
enum siglip_model_type {
    SIGLIP_MODEL_VIT_B_16,      // ViT-Base, Patch 16, 86M params
    SIGLIP_MODEL_VIT_L_16,      // ViT-Large, Patch 16, 303M params
    SIGLIP_MODEL_VIT_SO400M,    // ViT-SO400M, Patch 14, 400M params
    SIGLIP_MODEL_UNKNOWN
};

/**
 * Backend-Typ für Compute
 */
enum siglip_backend {
    SIGLIP_BACKEND_CPU,         // CPU (GGML)
    SIGLIP_BACKEND_CUDA,        // NVIDIA CUDA
    SIGLIP_BACKEND_METAL,       // Apple Metal
    SIGLIP_BACKEND_VULKAN,      // Vulkan (experimentell)
};

/**
 * Log-Level
 */
enum siglip_log_level {
    SIGLIP_LOG_NONE  = 0,
    SIGLIP_LOG_ERROR = 1,
    SIGLIP_LOG_WARN  = 2,
    SIGLIP_LOG_INFO  = 3,
    SIGLIP_LOG_DEBUG = 4,
};

/**
 * Embedding-Format für Output
 */
enum siglip_embed_format {
    SIGLIP_EMBED_F32,           // float32 Array
    SIGLIP_EMBED_F16,           // float16 Array
    SIGLIP_EMBED_NORMALIZED,    // L2-normalisiert
};

/**
 * Bild-Struktur
 */
struct siglip_image {
    uint8_t * data;             // RGB Pixel-Daten (HWC Format)
    int       width;            // Bildbreite
    int       height;           // Bildhöhe
    int       channels;         // Anzahl Kanäle (3 für RGB)
};

/**
 * Preprocessing-Parameter
 */
struct siglip_preprocess_params {
    int   target_size;          // Zielgröße (quadratisch)
    float mean[3];              // Normalisierungs-Mittelwert (RGB)
    float std[3];               // Normalisierungs-Standardabweichung (RGB)
    bool  center_crop;          // Center-Crop vor Resize
    bool  bicubic;              // Bicubic statt Bilinear Interpolation
};

/**
 * Modell-Hyperparameter
 */
struct siglip_hparams {
    enum siglip_model_type model_type;

    int hidden_size;            // Embedding-Dimension (768, 1024)
    int intermediate_size;      // MLP Hidden Size
    int num_attention_heads;    // Attention Heads
    int num_hidden_layers;      // Transformer Layers
    int image_size;             // Input-Bildgröße (224, 256, 384)
    int patch_size;             // Patch-Größe (14, 16)
    int num_patches;            // Anzahl Patches

    float layer_norm_eps;       // LayerNorm Epsilon

    struct siglip_preprocess_params preprocess;
};

/**
 * Inference-Parameter
 */
struct siglip_params {
    enum siglip_backend backend;
    enum siglip_log_level log_level;
    enum siglip_embed_format embed_format;

    int   n_threads;            // CPU Threads
    int   n_gpu_layers;         // GPU Layers (-1 = alle)
    int   main_gpu;             // Haupt-GPU Index
    bool  use_mmap;             // Memory-Mapping für Modell
    bool  use_mlock;            // Lock Memory

    // Batch-Processing
    int   batch_size;           // Bilder pro Batch
};

/**
 * Embedding-Resultat
 */
struct siglip_embedding {
    float * data;               // Embedding-Daten
    int     size;               // Embedding-Dimension
    int     batch_size;         // Anzahl Embeddings (bei Batch)
    bool    normalized;         // L2-normalisiert?
};

/**
 * Batch-Input
 */
struct siglip_batch {
    struct siglip_image ** images;
    int n_images;
};

/**
 * Kontext (opak)
 */
struct siglip_ctx;

/**
 * Callback für Progress
 */
typedef void (*siglip_progress_callback)(float progress, void * user_data);

// ============================================================================
// Funktionen: Initialisierung
// ============================================================================

/**
 * Erstellt Standard-Parameter
 */
struct siglip_params siglip_params_default(void);

/**
 * Lädt Modell aus GGUF-Datei
 *
 * @param model_path  Pfad zur GGUF-Datei
 * @param params      Inference-Parameter
 * @return            Kontext oder NULL bei Fehler
 */
struct siglip_ctx * siglip_load_model(
    const char * model_path,
    struct siglip_params params
);

/**
 * Lädt Modell mit Progress-Callback
 */
struct siglip_ctx * siglip_load_model_with_progress(
    const char * model_path,
    struct siglip_params params,
    siglip_progress_callback callback,
    void * user_data
);

/**
 * Gibt Kontext frei
 */
void siglip_free(struct siglip_ctx * ctx);

// ============================================================================
// Funktionen: Modell-Info
// ============================================================================

/**
 * Gibt Hyperparameter zurück
 */
const struct siglip_hparams * siglip_get_hparams(const struct siglip_ctx * ctx);

/**
 * Gibt Embedding-Dimension zurück
 */
int siglip_get_embedding_dim(const struct siglip_ctx * ctx);

/**
 * Gibt erwartete Bildgröße zurück
 */
int siglip_get_image_size(const struct siglip_ctx * ctx);

/**
 * Gibt Modell-Typ zurück
 */
enum siglip_model_type siglip_get_model_type(const struct siglip_ctx * ctx);

/**
 * Gibt Modell-Namen zurück
 */
const char * siglip_get_model_name(const struct siglip_ctx * ctx);

// ============================================================================
// Funktionen: Bild-Handling
// ============================================================================

/**
 * Lädt Bild aus Datei (JPG, PNG, BMP, etc.)
 *
 * @param path  Dateipfad
 * @return      Bild-Struktur oder NULL bei Fehler
 */
struct siglip_image * siglip_image_load(const char * path);

/**
 * Erstellt Bild aus Raw-Daten
 *
 * @param data      RGB Pixel-Daten (HWC Format)
 * @param width     Bildbreite
 * @param height    Bildhöhe
 * @param channels  Anzahl Kanäle (3 für RGB)
 * @return          Bild-Struktur oder NULL bei Fehler
 */
struct siglip_image * siglip_image_from_raw(
    const uint8_t * data,
    int width,
    int height,
    int channels
);

/**
 * Erstellt Bild aus Base64-String
 */
struct siglip_image * siglip_image_from_base64(const char * base64_data);

/**
 * Gibt Bild-Struktur frei
 */
void siglip_image_free(struct siglip_image * img);

/**
 * Klont ein Bild
 */
struct siglip_image * siglip_image_clone(const struct siglip_image * img);

// ============================================================================
// Funktionen: Preprocessing
// ============================================================================

/**
 * Preprocessing für ein Bild (Resize, Normalize)
 *
 * @param ctx  Kontext
 * @param img  Input-Bild
 * @return     Preprocessed float-Array [C, H, W] oder NULL bei Fehler
 */
float * siglip_preprocess(
    const struct siglip_ctx * ctx,
    const struct siglip_image * img
);

/**
 * Preprocessing mit benutzerdefinierten Parametern
 */
float * siglip_preprocess_with_params(
    const struct siglip_image * img,
    const struct siglip_preprocess_params * params
);

/**
 * Gibt preprocessed Daten frei
 */
void siglip_preprocess_free(float * preprocessed);

// ============================================================================
// Funktionen: Encoding
// ============================================================================

/**
 * Generiert Embedding für ein Bild
 *
 * @param ctx  Kontext
 * @param img  Input-Bild
 * @return     Embedding oder NULL bei Fehler
 */
struct siglip_embedding * siglip_encode(
    struct siglip_ctx * ctx,
    const struct siglip_image * img
);

/**
 * Generiert Embeddings für mehrere Bilder (Batch)
 *
 * @param ctx    Kontext
 * @param batch  Batch mit Bildern
 * @return       Embedding-Array oder NULL bei Fehler
 */
struct siglip_embedding * siglip_encode_batch(
    struct siglip_ctx * ctx,
    const struct siglip_batch * batch
);

/**
 * Generiert Embedding aus bereits preprocessed Daten
 *
 * @param ctx          Kontext
 * @param preprocessed Float-Array [C, H, W]
 * @return             Embedding oder NULL bei Fehler
 */
struct siglip_embedding * siglip_encode_preprocessed(
    struct siglip_ctx * ctx,
    const float * preprocessed
);

/**
 * Gibt Embedding-Struktur frei
 */
void siglip_embedding_free(struct siglip_embedding * emb);

// ============================================================================
// Funktionen: Utilities
// ============================================================================

/**
 * Berechnet Cosine Similarity zwischen zwei Embeddings
 */
float siglip_cosine_similarity(
    const struct siglip_embedding * a,
    const struct siglip_embedding * b
);

/**
 * Berechnet Cosine Similarity zwischen zwei Float-Arrays
 */
float siglip_cosine_similarity_raw(
    const float * a,
    const float * b,
    int size
);

/**
 * L2-Normalisierung eines Embeddings (in-place)
 */
void siglip_normalize(struct siglip_embedding * emb);

/**
 * L2-Normalisierung eines Float-Arrays (in-place)
 */
void siglip_normalize_raw(float * data, int size);

/**
 * Kopiert Embedding in Float-Array
 *
 * @param emb   Embedding
 * @param out   Output-Array (muss ausreichend Platz haben)
 * @param size  Array-Größe
 * @return      Anzahl kopierter Elemente
 */
int siglip_embedding_to_float(
    const struct siglip_embedding * emb,
    float * out,
    int size
);

// ============================================================================
// Funktionen: Serialisierung
// ============================================================================

/**
 * Serialisiert Embedding zu JSON
 *
 * @param emb  Embedding
 * @return     JSON-String (muss mit free() freigegeben werden)
 */
char * siglip_embedding_to_json(const struct siglip_embedding * emb);

/**
 * Serialisiert Embedding zu Binary (little-endian float32)
 *
 * @param emb       Embedding
 * @param out_size  Output: Größe in Bytes
 * @return          Binary-Daten (muss mit free() freigegeben werden)
 */
uint8_t * siglip_embedding_to_binary(
    const struct siglip_embedding * emb,
    size_t * out_size
);

/**
 * Serialisiert Embedding zu NumPy-kompatiblem Format
 */
uint8_t * siglip_embedding_to_numpy(
    const struct siglip_embedding * emb,
    size_t * out_size
);

// ============================================================================
// Funktionen: Fehlerbehandlung
// ============================================================================

/**
 * Gibt letzten Fehler zurück
 */
const char * siglip_get_last_error(void);

/**
 * Löscht letzten Fehler
 */
void siglip_clear_error(void);

/**
 * Setzt Log-Level
 */
void siglip_set_log_level(enum siglip_log_level level);

/**
 * Setzt Log-Callback
 */
typedef void (*siglip_log_callback)(enum siglip_log_level level, const char * msg, void * user_data);
void siglip_set_log_callback(siglip_log_callback callback, void * user_data);

// ============================================================================
// Funktionen: System-Info
// ============================================================================

/**
 * Gibt Version zurück
 */
const char * siglip_version(void);

/**
 * Gibt Build-Info zurück (Compiler, Flags, etc.)
 */
const char * siglip_build_info(void);

/**
 * Prüft ob Backend verfügbar ist
 */
bool siglip_backend_available(enum siglip_backend backend);

/**
 * Gibt verfügbare Backends zurück
 */
int siglip_get_available_backends(enum siglip_backend * backends, int max_backends);

/**
 * Gibt System-Info zurück (CPU Features, GPU, etc.)
 */
const char * siglip_system_info(void);

#ifdef __cplusplus
}
#endif

#endif // SIGLIP_H
