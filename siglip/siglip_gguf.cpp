/**
 * siglip_gguf.cpp - GGUF Parsing und Tensor-Loading
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - GGUF-Format Typen und Konstanten
 * - String/Metadaten lesen aus GGUF
 * - Tensor-Info Parsing
 * - Tensor-Zuweisung zu siglip_ctx
 */

#include "siglip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// GGML Headers
#include "ggml.h"
#include "ggml-alloc.h"

// ============================================================================
// GGUF Konstanten und Typen
// ============================================================================

// GGUF Magic Number und Version
constexpr uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION_MIN = 2;
constexpr uint32_t GGUF_VERSION_MAX = 3;

// GGUF Metadaten-Typen
enum gguf_metadata_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Tensor-Info Struktur fuer temporaere Speicherung beim Parsing
struct gguf_tensor_info {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
};

// ============================================================================
// GGUF String/Metadaten Lesen
// ============================================================================

/**
 * Liest einen GGUF-String (uint64 Laenge + Daten)
 */
std::string gguf_read_string(FILE * f) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return "";

    std::string s(len, '\0');
    if (fread(&s[0], 1, len, f) != len) return "";

    return s;
}

/**
 * Liest einen Metadaten-Wert basierend auf Typ
 *
 * @param f Datei-Handle
 * @param type GGUF-Typ
 * @param out Ausgabe-Buffer
 * @param max_size Maximale Groesse (fuer Strings)
 * @return true bei Erfolg
 */
bool gguf_read_metadata_value(FILE * f, uint32_t type, void * out, size_t max_size) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return fread(out, 1, 1, f) == 1;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return fread(out, 2, 1, f) == 1;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return fread(out, 4, 1, f) == 1;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return fread(out, 8, 1, f) == 1;

        case GGUF_TYPE_STRING: {
            std::string * str = static_cast<std::string *>(out);
            *str = gguf_read_string(f);
            return !str->empty() || feof(f) == 0;
        }

        default:
            return false;
    }
}

/**
 * Ueberspringt einen Metadaten-Wert
 */
bool gguf_skip_metadata_value(FILE * f, uint32_t type) {
    static const size_t type_sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

    if (type == GGUF_TYPE_STRING) {
        gguf_read_string(f);
        return true;
    }

    if (type == GGUF_TYPE_ARRAY) {
        uint32_t arr_type;
        uint64_t arr_len;
        fread(&arr_type, sizeof(arr_type), 1, f);
        fread(&arr_len, sizeof(arr_len), 1, f);

        for (uint64_t i = 0; i < arr_len; i++) {
            gguf_skip_metadata_value(f, arr_type);
        }
        return true;
    }

    if (type < sizeof(type_sizes) / sizeof(type_sizes[0])) {
        char buf[8];
        return fread(buf, type_sizes[type], 1, f) == 1;
    }

    return false;
}

// ============================================================================
// Tensor-Info Parsing
// ============================================================================

/**
 * Liest alle Tensor-Infos aus dem GGUF Header
 *
 * @param f Datei-Handle (positioniert nach Metadaten)
 * @param n_tensors Anzahl der Tensoren
 * @param out_infos Ausgabe-Vektor fuer Tensor-Infos
 * @return true bei Erfolg
 */
bool gguf_read_tensor_infos(FILE * f, uint64_t n_tensors,
                            std::vector<gguf_tensor_info> & out_infos) {
    out_infos.resize(n_tensors);

    for (uint64_t i = 0; i < n_tensors; i++) {
        out_infos[i].name = gguf_read_string(f);

        if (fread(&out_infos[i].n_dims, sizeof(uint32_t), 1, f) != 1) {
            return false;
        }

        out_infos[i].dims.resize(out_infos[i].n_dims);
        for (uint32_t j = 0; j < out_infos[i].n_dims; j++) {
            if (fread(&out_infos[i].dims[j], sizeof(uint64_t), 1, f) != 1) {
                return false;
            }
        }

        if (fread(&out_infos[i].type, sizeof(uint32_t), 1, f) != 1) {
            return false;
        }
        if (fread(&out_infos[i].offset, sizeof(uint64_t), 1, f) != 1) {
            return false;
        }
    }

    return true;
}

/**
 * Berechnet die Gesamtgroesse aller Tensoren in Bytes
 */
size_t gguf_calculate_tensor_size(const std::vector<gguf_tensor_info> & infos) {
    size_t total = 0;

    for (const auto & ti : infos) {
        size_t n_elements = 1;
        for (auto d : ti.dims) {
            n_elements *= d;
        }

        // Element-Groesse basierend auf Typ
        size_t element_size = 4; // Default F32
        switch (ti.type) {
            case GGML_TYPE_F16:  element_size = 2; break;
            case GGML_TYPE_Q8_0: element_size = 1; break; // Approximation
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1: element_size = 1; break; // Approximation
            default: element_size = 4; break;
        }

        total += n_elements * element_size;
    }

    return total;
}

// ============================================================================
// GGUF Header Validierung
// ============================================================================

/**
 * Validiert und liest den GGUF Header
 *
 * @param f Datei-Handle (am Anfang)
 * @param out_n_tensors Ausgabe: Anzahl Tensoren
 * @param out_n_kv Ausgabe: Anzahl Key-Value Paare
 * @param out_version Ausgabe: GGUF Version
 * @return true bei gueltigem Header
 */
bool gguf_validate_header(FILE * f, uint64_t * out_n_tensors,
                          uint64_t * out_n_kv, uint32_t * out_version) {
    // Magic pruefen
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GGUF_MAGIC) {
        return false;
    }

    // Version pruefen
    uint32_t version;
    if (fread(&version, sizeof(version), 1, f) != 1) {
        return false;
    }
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        return false;
    }

    // Tensor/KV Anzahl lesen
    if (fread(out_n_tensors, sizeof(uint64_t), 1, f) != 1) {
        return false;
    }
    if (fread(out_n_kv, sizeof(uint64_t), 1, f) != 1) {
        return false;
    }

    *out_version = version;
    return true;
}

/**
 * Berechnet das Data-Alignment nach dem Header
 *
 * @param f Datei-Handle (nach Header + Metadaten + Tensor-Infos)
 * @param alignment Standard-Alignment (32 bytes)
 * @return Position des Daten-Starts
 */
long gguf_get_data_start(FILE * f, int alignment) {
    long current_pos = ftell(f);
    long padding = (alignment - (current_pos % alignment)) % alignment;
    return current_pos + padding;
}
