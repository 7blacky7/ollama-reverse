/**
 * MODUL: clip_wrapper
 * ZWECK: C-Header fuer CLIP Vision Encoder Bindings (clip.cpp Wrapper)
 * INPUT: Modell-Pfad, Bild-Daten, Konfigurationsparameter
 * OUTPUT: Embedding-Vektoren, Modell-Metadaten
 * NEBENEFFEKTE: Speicherallokation, Modell-Laden
 * ABHAENGIGKEITEN: clip.cpp (extern), ggml (extern)
 * HINWEISE: Thread-sicher, Speicher muss mit clip_wrapper_free() freigegeben werden
 */

#ifndef CLIP_WRAPPER_H
#define CLIP_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Typen - Interne Strukturen
 * ============================================================================ */

// Opaker Context-Typ fuer CLIP-Modell
typedef struct clip_ctx clip_ctx;

/* ============================================================================
 * Initialisierungs-Parameter
 * ============================================================================ */

// Parameter fuer clip_init()
typedef struct clip_init_params {
    int32_t n_threads;      // Anzahl CPU-Threads
    int32_t n_gpu_layers;   // GPU-Layers (-1 = alle)
    int32_t main_gpu;       // Haupt-GPU Index
    int8_t  use_mmap;       // Memory-Mapping (0/1)
    int8_t  use_mlock;      // Memory-Locking (0/1)
} clip_init_params;

/* ============================================================================
 * Modell-Informationen
 * ============================================================================ */

// Modell-Metadaten
typedef struct clip_model_info {
    const char* name;       // Modell-Name
    int32_t embedding_dim;  // Embedding-Dimension
    int32_t image_size;     // Erwartete Bildgroesse (quadratisch)
} clip_model_info;

/* ============================================================================
 * Kern-Funktionen - Initialisierung
 * ============================================================================ */

// Default-Parameter holen
clip_init_params clip_wrapper_default_params(void);

// CLIP-Modell aus GGUF laden
// Rueckgabe: clip_ctx* bei Erfolg, NULL bei Fehler
clip_ctx* clip_wrapper_init(const char* model_path, clip_init_params params);

// CLIP-Context freigeben
void clip_wrapper_free(clip_ctx* ctx);

/* ============================================================================
 * Kern-Funktionen - Encoding
 * ============================================================================ */

// Einzelnes Bild zu Embedding konvertieren
// image_data: Rohbilddaten (JPEG, PNG, etc.)
// image_size: Groesse in Bytes
// embedding: Pre-allozierter Buffer fuer Ausgabe
// embedding_dim: Groesse des Embedding-Buffers
// Rueckgabe: 0 bei Erfolg, Fehlercode sonst
int clip_wrapper_encode_image(
    clip_ctx* ctx,
    const uint8_t* image_data,
    size_t image_size,
    float* embedding,
    int32_t embedding_dim
);

// Batch von Bildern zu Embeddings konvertieren
// images: Array von Bild-Pointern
// image_sizes: Array von Groessen
// batch_size: Anzahl Bilder
// embeddings: Pre-allozierter Buffer (batch_size * embedding_dim)
// Rueckgabe: 0 bei Erfolg, Fehlercode sonst
int clip_wrapper_encode_batch(
    clip_ctx* ctx,
    const uint8_t** images,
    const size_t* image_sizes,
    int32_t batch_size,
    float* embeddings,
    int32_t embedding_dim
);

/* ============================================================================
 * Hilfsfunktionen - Metadaten
 * ============================================================================ */

// Modell-Informationen abrufen
clip_model_info clip_wrapper_get_model_info(clip_ctx* ctx);

// Embedding-Dimension abrufen
int32_t clip_wrapper_get_embedding_dim(clip_ctx* ctx);

// Erwartete Bildgroesse abrufen
int32_t clip_wrapper_get_image_size(clip_ctx* ctx);

/* ============================================================================
 * Fehler-Codes
 * ============================================================================ */

#define CLIP_OK                0
#define CLIP_ERR_NULL_CTX     -1
#define CLIP_ERR_NULL_IMAGE   -2
#define CLIP_ERR_DECODE       -3
#define CLIP_ERR_ENCODE       -4
#define CLIP_ERR_ALLOC        -5

#ifdef __cplusplus
}
#endif

#endif /* CLIP_WRAPPER_H */
