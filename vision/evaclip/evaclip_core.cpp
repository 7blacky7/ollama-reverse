/**
 * MODUL: evaclip_core.cpp
 * ZWECK: EVA-CLIP Kontext-Verwaltung und GGUF Model Loading
 * INPUT: Modellpfad, Thread-Anzahl, Init-Parameter
 * OUTPUT: evaclip_ctx Instanz, Fehler-Strings
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O, Logging
 * ABHAENGIGKEITEN: evaclip_internal.h, ggml (extern)
 * HINWEISE: Thread-lokale Fehlerbehandlung, GGUF v2/v3 kompatibel
 *
 * EVA-CLIP Besonderheiten:
 * - EVA-Initialisierung aus Masked Autoencoder Pre-Training
 * - Bessere Skalierung bei grossen Modellen
 * - CLIP-kompatible Preprocessing-Parameter
 */

#include "evaclip_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[EVACLIP_MAX_ERROR_LEN] = {0};

// Log-Level (global)
evaclip_log_level g_evaclip_log_level = EVACLIP_LOG_INFO;

// ============================================================================
// Logging und Fehlerbehandlung
// ============================================================================

void evaclip_set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, EVACLIP_MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

void evaclip_log_msg(evaclip_log_level level, const char * fmt, ...) {
    // Level-Filter
    if (level > g_evaclip_log_level) return;

    // Nachricht formatieren
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // Prefix je nach Level
    const char * prefix = "";
    switch (level) {
        case EVACLIP_LOG_ERROR: prefix = "[ERROR] "; break;
        case EVACLIP_LOG_WARN:  prefix = "[WARN]  "; break;
        case EVACLIP_LOG_INFO:  prefix = "[INFO]  "; break;
        case EVACLIP_LOG_DEBUG: prefix = "[DEBUG] "; break;
        default: break;
    }
    fprintf(stderr, "evaclip: %s%s\n", prefix, buffer);
}

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

std::string evaclip_gguf_read_string(FILE * f) {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f) != 1) return "";

    std::string result(len, '\0');
    if (len > 0 && fread(&result[0], 1, len, f) != len) {
        return "";
    }
    return result;
}

bool evaclip_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size) {
    // Unterstuetzte GGUF-Typen
    constexpr uint32_t TYPE_UINT32 = 4;
    constexpr uint32_t TYPE_INT64  = 6;
    constexpr uint32_t TYPE_UINT64 = 10;
    constexpr uint32_t TYPE_STRING = 8;

    switch (type) {
        case TYPE_UINT32: {
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) return false;
            if (out && max_size >= sizeof(int64_t)) *(int64_t*)out = val;
            return true;
        }
        case TYPE_INT64:
        case TYPE_UINT64: {
            int64_t val;
            if (fread(&val, 8, 1, f) != 1) return false;
            if (out && max_size >= sizeof(int64_t)) *(int64_t*)out = val;
            return true;
        }
        case TYPE_STRING: {
            std::string str = evaclip_gguf_read_string(f);
            if (out) *(std::string*)out = str;
            return true;
        }
        default:
            return evaclip_gguf_skip_value(f, type);
    }
}

bool evaclip_gguf_skip_value(FILE * f, uint32_t type) {
    // Skip-Groessen fuer primitive Typen
    static const size_t skip_sizes[] = {1, 1, 2, 2, 4, 4, 8, 8, 0, 0, 8, 8, 8, 1};

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
// Hyperparameter Defaults setzen
// ============================================================================

static void init_default_hparams(evaclip_hparams * hp) {
    hp->hidden_size       = EVACLIP_DEFAULT_HIDDEN_SIZE;
    hp->intermediate_size = EVACLIP_DEFAULT_INTERMEDIATE;
    hp->num_attention_heads = EVACLIP_DEFAULT_HEADS;
    hp->num_hidden_layers = EVACLIP_DEFAULT_LAYERS;
    hp->image_size        = EVACLIP_DEFAULT_IMAGE_SIZE;
    hp->patch_size        = EVACLIP_DEFAULT_PATCH_SIZE;
    hp->num_patches       = (336 / 14) * (336 / 14);  // 576
    hp->layer_norm_eps    = 1e-6f;

    // CLIP-Standard Preprocessing (ImageNet normalisiert)
    hp->image_mean[0] = 0.48145466f;
    hp->image_mean[1] = 0.4578275f;
    hp->image_mean[2] = 0.40821073f;
    hp->image_std[0]  = 0.26862954f;
    hp->image_std[1]  = 0.26130258f;
    hp->image_std[2]  = 0.27577711f;
}

// ============================================================================
// Oeffentliche API - Default Parameter
// ============================================================================

evaclip_init_params evaclip_default_params(void) {
    evaclip_init_params params = {};
    params.n_threads = 0;       // Auto-detect
    params.n_gpu_layers = -1;   // Alle auf GPU
    params.main_gpu = 0;
    params.use_mmap = 1;
    params.use_mlock = 0;
    return params;
}

// ============================================================================
// Oeffentliche API - Modell laden
// ============================================================================

evaclip_ctx * evaclip_init(const char * model_path, evaclip_init_params params) {
    EVACLIP_LOG_INFO("Lade EVA-CLIP Modell: %s", model_path);

    // Datei oeffnen
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        evaclip_set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // GGUF Magic pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != EVACLIP_GGUF_MAGIC) {
        evaclip_set_error("Ungueltige GGUF Magic: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    EVACLIP_LOG_DEBUG("GGUF Version: %u", version);

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    EVACLIP_LOG_DEBUG("Tensoren: %lu, Metadaten: %lu", n_tensors, n_kv);

    // Kontext erstellen
    evaclip_ctx * ctx = new evaclip_ctx();
    ctx->model_path = model_path;
    ctx->n_threads = (params.n_threads > 0)
        ? params.n_threads
        : (int)std::thread::hardware_concurrency();

    // Default Hyperparameter
    init_default_hparams(&ctx->hparams);

    // Metadaten parsen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = evaclip_gguf_read_string(f);
        uint32_t type;
        fread(&type, 4, 1, f);

        // Relevante Keys extrahieren (evaclip.* oder vision.*)
        if (key == "general.name") {
            evaclip_gguf_read_value(f, type, &ctx->model_name, 0);
        } else if (key == "evaclip.hidden_size" || key == "vision.hidden_size") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "evaclip.num_attention_heads" || key == "vision.num_heads") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "evaclip.num_hidden_layers" || key == "vision.num_layers") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "evaclip.intermediate_size") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.intermediate_size = (int)val;
        } else if (key == "evaclip.image_size" || key == "vision.image_size") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.image_size = (int)val;
        } else if (key == "evaclip.patch_size" || key == "vision.patch_size") {
            int64_t val; evaclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.patch_size = (int)val;
        } else {
            evaclip_gguf_skip_value(f, type);
        }
    }

    // Patches berechnen
    int grid = ctx->hparams.image_size / ctx->hparams.patch_size;
    ctx->hparams.num_patches = grid * grid;

    EVACLIP_LOG_INFO("Modell: %s (Hidden: %d, Layers: %d, Patches: %d)",
                     ctx->model_name.c_str(), ctx->hparams.hidden_size,
                     ctx->hparams.num_hidden_layers, ctx->hparams.num_patches);

    // Tensoren laden
    if (!evaclip_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    EVACLIP_LOG_INFO("EVA-CLIP Modell erfolgreich geladen");
    return ctx;
}

// ============================================================================
// Oeffentliche API - Cleanup
// ============================================================================

void evaclip_free(evaclip_ctx * ctx) {
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
// Oeffentliche API - Modell-Info
// ============================================================================

evaclip_model_info evaclip_get_model_info(const evaclip_ctx * ctx) {
    evaclip_model_info info = {};
    if (ctx) {
        info.name = ctx->model_name.c_str();
        info.embedding_dim = ctx->hparams.hidden_size;
        info.image_size = ctx->hparams.image_size;
    }
    return info;
}
