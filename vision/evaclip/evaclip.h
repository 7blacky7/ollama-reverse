/**
 * MODUL: evaclip.h
 * ZWECK: C-Header fuer EVA-CLIP Vision Encoder Bindings
 * INPUT: Modell-Pfad, Bild-Daten, Parameter
 * OUTPUT: Embedding-Vektoren, Modell-Informationen
 * NEBENEFFEKTE: Speicherallokation fuer Modell und Embeddings
 * ABHAENGIGKEITEN: libevaclip (native Bibliothek)
 * HINWEISE: Muss mit libevaclip.a/.so/.dll gelinkt werden
 */

#ifndef EVACLIP_H
#define EVACLIP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque Typen
// ============================================================================

/** Opaque Handle fuer EVA-CLIP Context */
typedef struct evaclip_ctx evaclip_ctx;

// ============================================================================
// Initialisierungs-Parameter
// ============================================================================

/** Parameter fuer evaclip_init() */
typedef struct {
    int32_t n_threads;      // Anzahl CPU-Threads (0 = auto)
    int32_t n_gpu_layers;   // GPU-Layer (-1 = alle, 0 = keine)
    int32_t main_gpu;       // Haupt-GPU Index
    int8_t  use_mmap;       // Memory-Mapping aktivieren (1=ja, 0=nein)
    int8_t  use_mlock;      // Memory-Locking aktivieren (1=ja, 0=nein)
} evaclip_init_params;

// ============================================================================
// Modell-Informationen
// ============================================================================

/** Struktur fuer Modell-Metadaten */
typedef struct {
    const char* name;       // Modell-Name (z.B. "EVA02-CLIP-L-14")
    int32_t embedding_dim;  // Embedding-Dimension (z.B. 768, 1024)
    int32_t image_size;     // Erwartete Bildgroesse (z.B. 224, 336)
} evaclip_model_info;

// ============================================================================
// Initialisierung und Freigabe
// ============================================================================

/**
 * Gibt Default-Parameter zurueck.
 * Empfohlen: Immer zuerst aufrufen und dann anpassen.
 */
evaclip_init_params evaclip_default_params(void);

/**
 * Laedt ein EVA-CLIP Modell aus einer GGUF-Datei.
 *
 * @param model_path Pfad zur GGUF-Modelldatei
 * @param params Initialisierungs-Parameter
 * @return Context-Handle oder NULL bei Fehler
 */
evaclip_ctx* evaclip_init(const char* model_path, evaclip_init_params params);

/**
 * Gibt den Context und allen zugehoerigen Speicher frei.
 * Nach dem Aufruf ist ctx ungueltig.
 *
 * @param ctx Context-Handle
 */
void evaclip_free(evaclip_ctx* ctx);

// ============================================================================
// Modell-Informationen abrufen
// ============================================================================

/**
 * Gibt Metadaten ueber das geladene Modell zurueck.
 *
 * @param ctx Context-Handle
 * @return Modell-Informationen
 */
evaclip_model_info evaclip_get_model_info(const evaclip_ctx* ctx);

// ============================================================================
// Bild-Encoding
// ============================================================================

/**
 * Encodiert ein einzelnes Bild zu einem Embedding-Vektor.
 *
 * @param ctx Context-Handle
 * @param image_data Rohe Bild-Daten (JPEG/PNG)
 * @param image_size Groesse der Bild-Daten in Bytes
 * @param out_embedding Ausgabe-Buffer fuer Embedding (muss alloziert sein)
 * @param embedding_dim Dimension des Embeddings
 * @return 0 bei Erfolg, negativer Fehlercode bei Fehler
 *
 * Fehlercodes:
 *  -1: NULL context
 *  -2: NULL image data
 *  -3: Image decode failed
 *  -4: Encoding failed
 *  -5: Memory allocation failed
 */
int32_t evaclip_encode_image(
    evaclip_ctx* ctx,
    const uint8_t* image_data,
    size_t image_size,
    float* out_embedding,
    int32_t embedding_dim
);

/**
 * Encodiert mehrere Bilder zu Embedding-Vektoren (Batch-Verarbeitung).
 *
 * @param ctx Context-Handle
 * @param images Array von Bild-Daten-Pointern
 * @param image_sizes Array von Bild-Groessen
 * @param batch_size Anzahl der Bilder
 * @param out_embeddings Flaches Ausgabe-Array (batch_size * embedding_dim)
 * @param embedding_dim Dimension pro Embedding
 * @return 0 bei Erfolg, negativer Fehlercode bei Fehler
 */
int32_t evaclip_encode_batch(
    evaclip_ctx* ctx,
    const uint8_t** images,
    const size_t* image_sizes,
    int32_t batch_size,
    float* out_embeddings,
    int32_t embedding_dim
);

#ifdef __cplusplus
}
#endif

#endif /* EVACLIP_H */
