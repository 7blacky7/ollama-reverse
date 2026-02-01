/**
 * MODUL: nomic_core.cpp
 * ZWECK: Nomic Vision Kontext-Verwaltung und GGUF Model Loading
 * INPUT: Modellpfad, Thread-Anzahl
 * OUTPUT: nomic_ctx Instanz, Fehler-Strings
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O, Logging
 * ABHAENGIGKEITEN: nomic_internal.h, ggml (extern)
 * HINWEISE: Thread-lokale Fehlerbehandlung, GGUF v2/v3 kompatibel
 */

#include "nomic_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[NOMIC_MAX_ERROR_LEN] = {0};

// Log-Level (global)
nomic_log_level g_nomic_log_level = NOMIC_LOG_INFO;

// ============================================================================
// Logging und Fehlerbehandlung
// ============================================================================

void nomic_set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, NOMIC_MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

void nomic_log_msg(nomic_log_level level, const char * fmt, ...) {
    // Level-Filter
    if (level > g_nomic_log_level) return;

    // Nachricht formatieren
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // Prefix je nach Level
    const char * prefix = "";
    switch (level) {
        case NOMIC_LOG_ERROR: prefix = "[ERROR] "; break;
        case NOMIC_LOG_WARN:  prefix = "[WARN]  "; break;
        case NOMIC_LOG_INFO:  prefix = "[INFO]  "; break;
        case NOMIC_LOG_DEBUG: prefix = "[DEBUG] "; break;
        default: break;
    }
    fprintf(stderr, "nomic: %s%s\n", prefix, buffer);
}

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

std::string nomic_gguf_read_string(FILE * f) {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f) != 1) return "";

    std::string result(len, '\0');
    if (len > 0 && fread(&result[0], 1, len, f) != len) {
        return "";
    }
    return result;
}

bool nomic_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size) {
    // Unterstuetzte GGUF-Typen
    constexpr uint32_t TYPE_UINT32 = 4;
    constexpr uint32_t TYPE_INT64  = 6;
    constexpr uint32_t TYPE_STRING = 8;

    switch (type) {
        case TYPE_UINT32: {
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) return false;
            if (out && max_size >= sizeof(int64_t)) *(int64_t*)out = val;
            return true;
        }
        case TYPE_INT64: {
            int64_t val;
            if (fread(&val, 8, 1, f) != 1) return false;
            if (out && max_size >= sizeof(int64_t)) *(int64_t*)out = val;
            return true;
        }
        case TYPE_STRING: {
            std::string str = nomic_gguf_read_string(f);
            if (out) *(std::string*)out = str;
            return true;
        }
        default:
            return nomic_gguf_skip_value(f, type);
    }
}

bool nomic_gguf_skip_value(FILE * f, uint32_t type) {
    // Skip-Groessen fuer primitive Typen
    static const size_t skip_sizes[] = {1, 1, 2, 2, 4, 4, 8, 8, 0, 0, 0, 0, 8, 1};

    if (type == 8) { // String
        uint64_t len;
        if (fread(&len, 8, 1, f) != 1) return false;
        return fseek(f, (long)len, SEEK_CUR) == 0;
    }
    if (type < 14 && skip_sizes[type] > 0) {
        return fseek(f, (long)skip_sizes[type], SEEK_CUR) == 0;
    }
    return false;
}

// ============================================================================
// Modell laden
// ============================================================================

nomic_ctx * nomic_load_model(const char * model_path, int n_threads) {
    NOMIC_LOG_INFO("Lade Modell: %s", model_path);

    // Datei oeffnen
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        nomic_set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // GGUF Magic pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != NOMIC_GGUF_MAGIC) {
        nomic_set_error("Ungueltige GGUF Magic: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    NOMIC_LOG_DEBUG("GGUF Version: %u", version);

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    NOMIC_LOG_DEBUG("Tensoren: %lu, Metadaten: %lu", n_tensors, n_kv);

    // Kontext erstellen mit Default-Werten
    nomic_ctx * ctx = new nomic_ctx();
    ctx->model_path = model_path;
    ctx->n_threads = (n_threads > 0) ? n_threads : (int)std::thread::hardware_concurrency();

    // Default Hyperparameter fuer Nomic Embed Vision
    ctx->hparams.hidden_size       = NOMIC_DEFAULT_HIDDEN_SIZE;
    ctx->hparams.intermediate_size = 3072;
    ctx->hparams.num_attention_heads = NOMIC_MAX_HEADS;
    ctx->hparams.num_hidden_layers = NOMIC_MAX_LAYERS;
    ctx->hparams.image_size        = NOMIC_DEFAULT_IMAGE_SIZE;
    ctx->hparams.patch_size        = NOMIC_DEFAULT_PATCH_SIZE;
    ctx->hparams.num_patches       = (384 / 14) * (384 / 14);  // 729
    ctx->hparams.layer_norm_eps    = 1e-6f;

    // Metadaten parsen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = nomic_gguf_read_string(f);
        uint32_t type;
        fread(&type, 4, 1, f);

        // Relevante Keys extrahieren
        if (key == "general.name") {
            nomic_gguf_read_value(f, type, &ctx->model_name, 0);
        } else if (key == "nomic.hidden_size" || key == "vision.hidden_size") {
            int64_t val; nomic_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "nomic.num_attention_heads" || key == "vision.num_heads") {
            int64_t val; nomic_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "nomic.num_hidden_layers" || key == "vision.num_layers") {
            int64_t val; nomic_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "nomic.image_size" || key == "vision.image_size") {
            int64_t val; nomic_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.image_size = (int)val;
        } else if (key == "nomic.patch_size" || key == "vision.patch_size") {
            int64_t val; nomic_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.patch_size = (int)val;
        } else {
            nomic_gguf_skip_value(f, type);
        }
    }

    // Patches berechnen
    int grid = ctx->hparams.image_size / ctx->hparams.patch_size;
    ctx->hparams.num_patches = grid * grid;

    NOMIC_LOG_INFO("Modell: %s (Hidden: %d, Layers: %d, Patches: %d)",
                   ctx->model_name.c_str(), ctx->hparams.hidden_size,
                   ctx->hparams.num_hidden_layers, ctx->hparams.num_patches);

    // Tensoren laden (Placeholder - echte Impl in separater Datei)
    if (!nomic_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    NOMIC_LOG_INFO("Modell erfolgreich geladen");
    return ctx;
}

// ============================================================================
// Cleanup
// ============================================================================

void nomic_free(nomic_ctx * ctx) {
    if (!ctx) return;

    // GGML Ressourcen freigeben
    if (ctx->allocr)     ggml_gallocr_free(ctx->allocr);
    if (ctx->buffer)     ggml_backend_buffer_free(ctx->buffer);
    if (ctx->backend)    ggml_backend_free(ctx->backend);
    if (ctx->ctx_compute) ggml_free(ctx->ctx_compute);
    if (ctx->ctx_data)   ggml_free(ctx->ctx_data);

    delete ctx;
}

// ============================================================================
// Info-Abfragen
// ============================================================================

int nomic_get_embedding_dim(const nomic_ctx * ctx) {
    return ctx ? ctx->hparams.hidden_size : 0;
}

int nomic_get_image_size(const nomic_ctx * ctx) {
    return ctx ? ctx->hparams.image_size : 0;
}

const nomic_hparams * nomic_get_hparams(const nomic_ctx * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ============================================================================
// Fehler-API
// ============================================================================

const char * nomic_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}

void nomic_clear_error(void) {
    g_last_error[0] = '\0';
}

void nomic_set_log_level(nomic_log_level level) {
    g_nomic_log_level = level;
}
