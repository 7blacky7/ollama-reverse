/**
 * siglip_serialize.cpp - SigLIP Embedding Serialisierung
 *
 * Dieser Teil implementiert die Serialisierung von Embeddings:
 * - siglip_embedding_to_json() - JSON Format
 * - siglip_embedding_to_binary() - Rohe float32 Daten
 * - siglip_embedding_to_numpy() - NumPy .npy Format
 */

#include "siglip.h"

#include <cstdio>
#include <cstring>
#include <string>

// ============================================================================
// Oeffentliche API - Serialisierung
// ============================================================================

/**
 * Konvertiert ein Embedding zu JSON-String
 *
 * Format: {"embedding":[...], "size":768, "normalized":false}
 *
 * @param emb Das zu serialisierende Embedding
 * @return JSON-String (muss mit delete[] freigegeben werden)
 */
char * siglip_embedding_to_json(const siglip_embedding * emb) {
    if (!emb) return nullptr;

    // JSON String aufbauen
    std::string json = "{\"embedding\":[";

    // Embedding-Werte hinzufuegen
    for (int i = 0; i < emb->size; i++) {
        if (i > 0) json += ",";
        char buf[32];
        snprintf(buf, sizeof(buf), "%.6f", emb->data[i]);
        json += buf;
    }

    json += "],\"size\":";
    json += std::to_string(emb->size);
    json += ",\"normalized\":";
    json += emb->normalized ? "true" : "false";
    json += "}";

    // C-String Kopie erstellen
    char * result = new char[json.size() + 1];
    strcpy(result, json.c_str());
    return result;
}

/**
 * Konvertiert ein Embedding zu rohen Binaerdaten (float32)
 *
 * @param emb Das zu serialisierende Embedding
 * @param out_size Ausgabe: Groesse in Bytes
 * @return Binaer-Daten (muss mit delete[] freigegeben werden)
 */
uint8_t * siglip_embedding_to_binary(const siglip_embedding * emb, size_t * out_size) {
    if (!emb || !out_size) return nullptr;

    *out_size = emb->size * sizeof(float);
    uint8_t * result = new uint8_t[*out_size];
    memcpy(result, emb->data, *out_size);
    return result;
}

/**
 * Konvertiert ein Embedding zu NumPy .npy Format
 *
 * @param emb Das zu serialisierende Embedding
 * @param out_size Ausgabe: Groesse in Bytes
 * @return .npy Daten (muss mit delete[] freigegeben werden)
 */
uint8_t * siglip_embedding_to_numpy(const siglip_embedding * emb, size_t * out_size) {
    if (!emb || !out_size) return nullptr;

    // NumPy .npy Format (Version 1.0):
    // - Magic: \x93NUMPY
    // - Version: 1.0 (2 bytes)
    // - Header Length: uint16 (little-endian)
    // - Header: Python dict als ASCII String
    // - Padding auf 64-byte Alignment
    // - Daten

    // Header erstellen
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(emb->size);
    header += ",), }";

    // Padding auf 64 bytes (nach Magic + Version + Header Length)
    size_t header_len = header.size();
    size_t total_header_size = 10 + header_len; // 6 magic + 2 version + 2 length
    size_t padding = (64 - (total_header_size % 64)) % 64;
    if (padding == 0) padding = 64; // Mindestens ein Padding fuer newline

    // Padding und Newline hinzufuegen
    header.append(padding - 1, ' ');
    header += '\n';

    // Gesamtgroesse berechnen
    size_t total_size = 10 + header.size() + emb->size * sizeof(float);
    uint8_t * result = new uint8_t[total_size];

    // ====================================
    // Magic Number schreiben
    // ====================================
    result[0] = 0x93;
    result[1] = 'N';
    result[2] = 'U';
    result[3] = 'M';
    result[4] = 'P';
    result[5] = 'Y';

    // Version 1.0
    result[6] = 1;
    result[7] = 0;

    // Header Length (little-endian uint16)
    uint16_t hlen = static_cast<uint16_t>(header.size());
    result[8] = hlen & 0xFF;
    result[9] = (hlen >> 8) & 0xFF;

    // Header kopieren
    memcpy(result + 10, header.c_str(), header.size());

    // Embedding-Daten kopieren
    memcpy(result + 10 + header.size(), emb->data, emb->size * sizeof(float));

    *out_size = total_size;
    return result;
}
