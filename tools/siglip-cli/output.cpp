/**
 * output.cpp - Output-Format Implementation
 *
 * Implementiert Export-Funktionen fuer:
 * - JSON: Menschenlesbar, mit Metadaten
 * - Binary: Kompakt, schnell zu laden
 * - NumPy: Kompatibel mit Python numpy.load()
 */

#include "output.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

// ============================================================================
// JSON Output
// ============================================================================

bool write_json(const siglip_embedding * emb, const std::string & path,
                const std::string & filename) {
    std::stringstream ss;
    ss << std::setprecision(8);

    ss << "{\n";
    if (!filename.empty()) {
        ss << "  \"file\": \"" << filename << "\",\n";
    }
    ss << "  \"dimension\": " << emb->size << ",\n";
    ss << "  \"normalized\": " << (emb->normalized ? "true" : "false") << ",\n";
    ss << "  \"embedding\": [";

    for (int i = 0; i < emb->size; i++) {
        if (i > 0) ss << ", ";
        if (i % 8 == 0) ss << "\n    ";
        ss << emb->data[i];
    }
    ss << "\n  ]\n}";

    // Ausgabe
    if (path.empty()) {
        printf("%s\n", ss.str().c_str());
        return true;
    }

    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }
    f << ss.str();
    return true;
}

bool write_json_batch(const std::vector<siglip_embedding *> & embeddings,
                      const std::vector<std::string> & filenames,
                      const std::string & path) {
    std::stringstream ss;
    ss << std::setprecision(8);
    ss << "[\n";

    for (size_t i = 0; i < embeddings.size(); i++) {
        if (i > 0) ss << ",\n";

        const siglip_embedding * emb = embeddings[i];
        ss << "  {\n";
        ss << "    \"file\": \"" << filenames[i] << "\",\n";
        ss << "    \"embedding\": [";

        for (int j = 0; j < emb->size; j++) {
            if (j > 0) ss << ", ";
            ss << emb->data[j];
        }
        ss << "]\n  }";
    }

    ss << "\n]";

    if (path.empty()) {
        printf("%s\n", ss.str().c_str());
        return true;
    }

    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }
    f << ss.str();
    return true;
}

// ============================================================================
// Binary Output
// ============================================================================

bool write_binary(const siglip_embedding * emb, const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char *>(emb->data), emb->size * sizeof(float));
    return true;
}

bool write_binary_batch(const std::vector<siglip_embedding *> & embeddings,
                        const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }

    for (const auto * emb : embeddings) {
        f.write(reinterpret_cast<const char *>(emb->data), emb->size * sizeof(float));
    }
    return true;
}

// ============================================================================
// NumPy Output
// ============================================================================

/**
 * Schreibt NumPy .npy Header (Version 1.0)
 */
static void write_npy_header(std::ofstream & f, int rows, int cols) {
    // Magic: \x93NUMPY
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    f.write(reinterpret_cast<const char *>(magic), 6);

    // Version 1.0
    f.put(0x01);
    f.put(0x00);

    // Header-String erstellen
    std::stringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    if (rows > 1) {
        header << rows << ", " << cols;
    } else {
        header << cols << ",";
    }
    header << "), }";

    // Padding auf 64-Byte Grenze (Header-Laenge + 10 Bytes fuer Magic+Version+Len)
    std::string h = header.str();
    while ((h.size() + 10) % 64 != 0) {
        h += ' ';
    }
    h += '\n';

    // Header-Laenge (little-endian uint16)
    uint16_t len = static_cast<uint16_t>(h.size());
    f.put(len & 0xFF);
    f.put((len >> 8) & 0xFF);

    // Header schreiben
    f.write(h.c_str(), h.size());
}

bool write_numpy(const siglip_embedding * emb, const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }

    write_npy_header(f, 1, emb->size);
    f.write(reinterpret_cast<const char *>(emb->data), emb->size * sizeof(float));
    return true;
}

bool write_numpy_batch(const std::vector<siglip_embedding *> & embeddings,
                       const std::string & path) {
    if (embeddings.empty()) return false;

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Fehler: Kann '%s' nicht oeffnen\n", path.c_str());
        return false;
    }

    int rows = static_cast<int>(embeddings.size());
    int cols = embeddings[0]->size;

    write_npy_header(f, rows, cols);

    for (const auto * emb : embeddings) {
        f.write(reinterpret_cast<const char *>(emb->data), emb->size * sizeof(float));
    }
    return true;
}
