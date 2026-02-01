/**
 * MODUL: dinov2_core.cpp
 * ZWECK: DINOv2 Kontext-Verwaltung und GGUF Model Loading
 * INPUT: Modellpfad, Thread-Anzahl
 * OUTPUT: dinov2_ctx Instanz, Fehler-Strings
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O, Logging
 * ABHAENGIGKEITEN: dinov2_internal.h, ggml (extern)
 * HINWEISE: Thread-lokale Fehlerbehandlung, GGUF v2/v3 kompatibel
 *
 * DINOv2 Varianten:
 * - dinov2-s: 384 hidden, 6 heads, 12 layers
 * - dinov2-b: 768 hidden, 12 heads, 12 layers
 * - dinov2-l: 1024 hidden, 16 heads, 24 layers
 * - dinov2-g: 1536 hidden, 24 heads, 40 layers
 */

#include "dinov2_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[DINOV2_MAX_ERROR_LEN] = {0};

// Log-Level (global)
dinov2_log_level g_dinov2_log_level = DINOV2_LOG_INFO;

// ============================================================================
// Logging und Fehlerbehandlung
// ============================================================================

void dinov2_set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, DINOV2_MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

void dinov2_log_msg(dinov2_log_level level, const char * fmt, ...) {
    // Level-Filter
    if (level > g_dinov2_log_level) return;

    // Nachricht formatieren
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // Prefix je nach Level
    const char * prefix = "";
    switch (level) {
        case DINOV2_LOG_ERROR: prefix = "[ERROR] "; break;
        case DINOV2_LOG_WARN:  prefix = "[WARN]  "; break;
        case DINOV2_LOG_INFO:  prefix = "[INFO]  "; break;
        case DINOV2_LOG_DEBUG: prefix = "[DEBUG] "; break;
        default: break;
    }
    fprintf(stderr, "dinov2: %s%s\n", prefix, buffer);
}

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

std::string dinov2_gguf_read_string(FILE * f) {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f) != 1) return "";

    std::string result(len, '\0');
    if (len > 0 && fread(&result[0], 1, len, f) != len) {
        return "";
    }
    return result;
}

bool dinov2_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size) {
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
            std::string str = dinov2_gguf_read_string(f);
            if (out) *(std::string*)out = str;
            return true;
        }
        default:
            return dinov2_gguf_skip_value(f, type);
    }
}

bool dinov2_gguf_skip_value(FILE * f, uint32_t type) {
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
// Forward Declarations
// ============================================================================

static void init_default_hparams(dinov2_hparams * hp);
static bool dinov2_parse_metadata(dinov2_ctx * ctx, FILE * f, uint64_t n_kv);

// ============================================================================
// Hyperparameter Defaults setzen
// ============================================================================

static void init_default_hparams(dinov2_hparams * hp) {
    // DINOv2-B Defaults (Base Variante)
    hp->hidden_size        = DINOV2_DEFAULT_HIDDEN_SIZE;
    hp->intermediate_size  = 3072;
    hp->num_attention_heads = DINOV2_MAX_HEADS;
    hp->num_hidden_layers  = DINOV2_MAX_LAYERS;
    hp->image_size         = DINOV2_DEFAULT_IMAGE_SIZE;
    hp->patch_size         = DINOV2_DEFAULT_PATCH_SIZE;
    hp->num_patches        = (518 / 14) * (518 / 14);  // 1369
    hp->layer_norm_eps     = 1e-6f;
}

// ============================================================================
// Oeffentliche API - Modell laden
// ============================================================================

dinov2_ctx * dinov2_load(const char * model_path, int n_threads) {
    DINOV2_LOG_INFO("Lade DINOv2 Modell: %s", model_path);

    // Datei oeffnen
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        dinov2_set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // GGUF Magic pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != DINOV2_GGUF_MAGIC) {
        dinov2_set_error("Ungueltige GGUF Magic: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    DINOV2_LOG_DEBUG("GGUF Version: %u", version);

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    DINOV2_LOG_DEBUG("Tensoren: %lu, Metadaten: %lu", n_tensors, n_kv);

    // Kontext erstellen
    dinov2_ctx * ctx = new dinov2_ctx();
    ctx->model_path = model_path;
    ctx->n_threads = (n_threads > 0)
        ? n_threads
        : (int)std::thread::hardware_concurrency();
    ctx->num_register_tokens = 0;  // Default: keine Register-Tokens

    // Default Hyperparameter
    init_default_hparams(&ctx->hparams);

    // Metadaten parsen (dinov2.* oder vision.*)
    if (!dinov2_parse_metadata(ctx, f, n_kv)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    // Patches berechnen
    int grid = ctx->hparams.image_size / ctx->hparams.patch_size;
    ctx->hparams.num_patches = grid * grid;

    DINOV2_LOG_INFO("Modell: %s (Hidden: %d, Layers: %d, Patches: %d)",
                    ctx->model_name.c_str(), ctx->hparams.hidden_size,
                    ctx->hparams.num_hidden_layers, ctx->hparams.num_patches);

    // Tensoren laden
    if (!dinov2_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    DINOV2_LOG_INFO("DINOv2 Modell erfolgreich geladen");
    return ctx;
}

// ============================================================================
// Metadaten parsen
// ============================================================================

static bool dinov2_parse_metadata(dinov2_ctx * ctx, FILE * f, uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = dinov2_gguf_read_string(f);
        uint32_t type;
        if (fread(&type, 4, 1, f) != 1) return false;

        // Relevante Keys extrahieren
        if (key == "general.name") {
            dinov2_gguf_read_value(f, type, &ctx->model_name, 0);
        } else if (key == "dinov2.hidden_size" || key == "vision.hidden_size") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "dinov2.num_attention_heads" || key == "vision.num_heads") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "dinov2.num_hidden_layers" || key == "vision.num_layers") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "dinov2.intermediate_size") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.intermediate_size = (int)val;
        } else if (key == "dinov2.image_size" || key == "vision.image_size") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.image_size = (int)val;
        } else if (key == "dinov2.patch_size" || key == "vision.patch_size") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.patch_size = (int)val;
        } else if (key == "dinov2.num_register_tokens") {
            int64_t val; dinov2_gguf_read_value(f, type, &val, sizeof(val));
            ctx->num_register_tokens = (int)val;
        } else {
            dinov2_gguf_skip_value(f, type);
        }
    }
    return true;
}

// ============================================================================
// Oeffentliche API - Cleanup
// ============================================================================

void dinov2_free(dinov2_ctx * ctx) {
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

int dinov2_get_dim(const dinov2_ctx * ctx) {
    return ctx ? ctx->hparams.hidden_size : 0;
}

int dinov2_get_num_patches(const dinov2_ctx * ctx) {
    return ctx ? ctx->hparams.num_patches : 0;
}

int dinov2_get_image_size(const dinov2_ctx * ctx) {
    return ctx ? ctx->hparams.image_size : 0;
}

const dinov2_hparams * dinov2_get_hparams(const dinov2_ctx * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ============================================================================
// Oeffentliche API - Fehlerbehandlung
// ============================================================================

const char * dinov2_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}

void dinov2_clear_error(void) {
    g_last_error[0] = '\0';
}

void dinov2_set_log_level(dinov2_log_level level) {
    g_dinov2_log_level = level;
}
