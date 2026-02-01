/**
 * siglip_inference.cpp - SigLIP Encoding und Embedding Utilities
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - Encoding (siglip_encode, siglip_encode_preprocessed, siglip_encode_batch)
 * - Embedding Speicherverwaltung (siglip_embedding_free)
 * - Similarity Berechnung (cosine_similarity, cosine_similarity_raw)
 * - Normalisierung (normalize, normalize_raw)
 * - Embedding Export (to_float)
 *
 * Transformer-Komponenten: siehe siglip_transformer.cpp
 * Serialisierung: siehe siglip_serialize.cpp
 */

#include "siglip.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// Forward-Deklarationen fuer interne Strukturen
// ============================================================================

// Interne Kontext-Struktur (vollstaendig in siglip_core.cpp definiert)
struct siglip_ctx;

// Zugriff auf interne Kontext-Felder (aus siglip_core.cpp)
extern const siglip_hparams * siglip_get_hparams(const siglip_ctx * ctx);

// ============================================================================
// Oeffentliche API - Encoding
// ============================================================================

/**
 * Encodiert ein Bild zu einem Embedding-Vektor
 *
 * @param ctx SigLIP Kontext
 * @param img Das zu encodierende Bild
 * @return Embedding oder nullptr bei Fehler
 */
siglip_embedding * siglip_encode(siglip_ctx * ctx, const siglip_image * img) {
    if (!ctx || !img) {
        return nullptr;
    }

    // Preprocessing durchfuehren
    float * preprocessed = siglip_preprocess(ctx, img);
    if (!preprocessed) {
        return nullptr;
    }

    // Encoding mit preprocessed Input
    siglip_embedding * result = siglip_encode_preprocessed(ctx, preprocessed);

    // Preprocessing-Speicher freigeben
    siglip_preprocess_free(preprocessed);

    return result;
}

/**
 * Encodiert bereits vorverarbeitete Bilddaten
 *
 * @param ctx SigLIP Kontext
 * @param preprocessed Float-Array im CHW Format [3, H, W]
 * @return Embedding oder nullptr bei Fehler
 */
siglip_embedding * siglip_encode_preprocessed(siglip_ctx * ctx, const float * preprocessed) {
    if (!ctx || !preprocessed) {
        return nullptr;
    }

    const siglip_hparams * hp = siglip_get_hparams(ctx);
    if (!hp) return nullptr;

    // Placeholder fuer die vollstaendige Graph-basierte Inferenz
    // Die echte Implementierung nutzt build_graph() und ggml_backend_graph_compute()

    siglip_embedding * emb = new siglip_embedding();
    emb->size = hp->hidden_size;
    emb->batch_size = 1;
    emb->normalized = false;
    emb->data = new float[hp->hidden_size];

    // Initialisiere mit Nullen (echte Inferenz wuerde hier Werte berechnen)
    memset(emb->data, 0, hp->hidden_size * sizeof(float));

    return emb;
}

/**
 * Encodiert einen Batch von Bildern
 *
 * @param ctx SigLIP Kontext
 * @param batch Batch-Struktur mit Bildern
 * @return Batch-Embedding oder nullptr bei Fehler
 */
siglip_embedding * siglip_encode_batch(siglip_ctx * ctx, const siglip_batch * batch) {
    if (!ctx || !batch || batch->n_images <= 0) {
        return nullptr;
    }

    const siglip_hparams * hp = siglip_get_hparams(ctx);
    if (!hp) return nullptr;

    // Batch-Embeddings sammeln
    siglip_embedding * result = new siglip_embedding();
    result->size = hp->hidden_size;
    result->batch_size = batch->n_images;
    result->normalized = false;
    result->data = new float[hp->hidden_size * batch->n_images];

    // Jedes Bild einzeln encodieren
    for (int i = 0; i < batch->n_images; i++) {
        siglip_embedding * single = siglip_encode(ctx, batch->images[i]);
        if (single) {
            memcpy(result->data + i * hp->hidden_size, single->data, hp->hidden_size * sizeof(float));
            siglip_embedding_free(single);
        } else {
            memset(result->data + i * hp->hidden_size, 0, hp->hidden_size * sizeof(float));
        }
    }

    return result;
}

/**
 * Gibt den Speicher eines Embeddings frei
 */
void siglip_embedding_free(siglip_embedding * emb) {
    if (emb) {
        delete[] emb->data;
        delete emb;
    }
}

// ============================================================================
// Oeffentliche API - Embedding Utilities
// ============================================================================

/**
 * Berechnet die Kosinus-Aehnlichkeit zwischen zwei Embeddings
 *
 * @param a Erstes Embedding
 * @param b Zweites Embedding
 * @return Aehnlichkeit im Bereich [-1, 1]
 */
float siglip_cosine_similarity(const siglip_embedding * a, const siglip_embedding * b) {
    if (!a || !b || a->size != b->size) return 0.0f;
    return siglip_cosine_similarity_raw(a->data, b->data, a->size);
}

/**
 * Berechnet die Kosinus-Aehnlichkeit zwischen zwei Float-Arrays
 *
 * Formel: cos(a, b) = (a . b) / (||a|| * ||b||)
 *
 * @param a Erstes Array
 * @param b Zweites Array
 * @param size Laenge der Arrays
 * @return Aehnlichkeit im Bereich [-1, 1]
 */
float siglip_cosine_similarity_raw(const float * a, const float * b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    // Division durch Null vermeiden
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;

    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

/**
 * Normalisiert ein Embedding auf Einheitslaenge (L2-Norm = 1)
 *
 * @param emb Das zu normalisierende Embedding (in-place)
 */
void siglip_normalize(siglip_embedding * emb) {
    if (!emb || !emb->data) return;
    siglip_normalize_raw(emb->data, emb->size * emb->batch_size);
    emb->normalized = true;
}

/**
 * Normalisiert ein Float-Array auf Einheitslaenge
 *
 * @param data Das zu normalisierende Array (in-place)
 * @param size Laenge des Arrays
 */
void siglip_normalize_raw(float * data, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        for (int i = 0; i < size; i++) {
            data[i] /= norm;
        }
    }
}

/**
 * Kopiert Embedding-Daten in ein externes Float-Array
 *
 * @param emb Das Quell-Embedding
 * @param out Ziel-Array
 * @param size Maximale Anzahl zu kopierender Elemente
 * @return Anzahl tatsaechlich kopierter Elemente
 */
int siglip_embedding_to_float(const siglip_embedding * emb, float * out, int size) {
    if (!emb || !out) return 0;
    int n = std::min(size, emb->size * emb->batch_size);
    memcpy(out, emb->data, n * sizeof(float));
    return n;
}
