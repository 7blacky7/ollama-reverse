/**
 * MODUL: openclip_core.cpp
 * ZWECK: OpenCLIP Kontext-Verwaltung und GGUF Model Loading
 * INPUT: Modellpfad, Thread-Anzahl
 * OUTPUT: openclip_ctx Instanz, Fehler-Strings
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O, Logging
 * ABHAENGIGKEITEN: openclip_internal.h, ggml (extern)
 * HINWEISE: Thread-lokale Fehlerbehandlung, GGUF v2/v3 kompatibel
 *           OpenCLIP Modelle nutzen andere Tensor-Namen als CLIP
 */

#include "openclip_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[OPENCLIP_MAX_ERROR_LEN] = {0};

// Log-Level (global)
openclip_log_level g_openclip_log_level = OPENCLIP_LOG_INFO;

// ============================================================================
// Logging und Fehlerbehandlung
// ============================================================================

void openclip_set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, OPENCLIP_MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

void openclip_log_msg(openclip_log_level level, const char * fmt, ...) {
    // Level-Filter
    if (level > g_openclip_log_level) return;

    // Nachricht formatieren
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // Prefix je nach Level
    const char * prefix = "";
    switch (level) {
        case OPENCLIP_LOG_ERROR: prefix = "[ERROR] "; break;
        case OPENCLIP_LOG_WARN:  prefix = "[WARN]  "; break;
        case OPENCLIP_LOG_INFO:  prefix = "[INFO]  "; break;
        case OPENCLIP_LOG_DEBUG: prefix = "[DEBUG] "; break;
        default: break;
    }
    fprintf(stderr, "openclip: %s%s\n", prefix, buffer);
}

// ============================================================================
// GGUF Hilfsfunktionen
// ============================================================================

std::string openclip_gguf_read_string(FILE * f) {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f) != 1) return "";

    std::string result(len, '\0');
    if (len > 0 && fread(&result[0], 1, len, f) != len) {
        return "";
    }
    return result;
}

bool openclip_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size) {
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
            std::string str = openclip_gguf_read_string(f);
            if (out) *(std::string*)out = str;
            return true;
        }
        default:
            return openclip_gguf_skip_value(f, type);
    }
}

bool openclip_gguf_skip_value(FILE * f, uint32_t type) {
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

openclip_ctx * openclip_load(const char * model_path, int n_threads) {
    OPENCLIP_LOG_INFO("Lade OpenCLIP Modell: %s", model_path);

    // Datei oeffnen
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        openclip_set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // GGUF Magic pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != OPENCLIP_GGUF_MAGIC) {
        openclip_set_error("Ungueltige GGUF Magic: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    OPENCLIP_LOG_DEBUG("GGUF Version: %u", version);

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    OPENCLIP_LOG_DEBUG("Tensoren: %lu, Metadaten: %lu", n_tensors, n_kv);

    // Kontext erstellen mit Default-Werten fuer ViT-bigG-14
    openclip_ctx * ctx = new openclip_ctx();
    ctx->model_path = model_path;
    ctx->n_threads = (n_threads > 0) ? n_threads : (int)std::thread::hardware_concurrency();

    // Default Hyperparameter fuer ViT-bigG-14 (groesstes OpenCLIP Modell)
    ctx->hparams.hidden_size       = OPENCLIP_DEFAULT_HIDDEN_SIZE;  // 1280
    ctx->hparams.intermediate_size = 5120;  // 4x hidden
    ctx->hparams.num_attention_heads = OPENCLIP_MAX_HEADS;  // 20
    ctx->hparams.num_hidden_layers = OPENCLIP_MAX_LAYERS;   // 40
    ctx->hparams.image_size        = OPENCLIP_DEFAULT_IMAGE_SIZE;  // 224
    ctx->hparams.patch_size        = OPENCLIP_DEFAULT_PATCH_SIZE;  // 14
    ctx->hparams.num_patches       = (224 / 14) * (224 / 14);  // 256
    ctx->hparams.layer_norm_eps    = 1e-5f;  // OpenCLIP nutzt 1e-5

    // Metadaten parsen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = openclip_gguf_read_string(f);
        uint32_t type;
        fread(&type, 4, 1, f);

        // Relevante Keys extrahieren (openclip.* oder vision.*)
        if (key == "general.name") {
            openclip_gguf_read_value(f, type, &ctx->model_name, 0);
        } else if (key == "openclip.hidden_size" || key == "vision.hidden_size") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "openclip.num_attention_heads" || key == "vision.num_heads") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "openclip.num_hidden_layers" || key == "vision.num_layers") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "openclip.intermediate_size" || key == "vision.intermediate_size") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.intermediate_size = (int)val;
        } else if (key == "openclip.image_size" || key == "vision.image_size") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.image_size = (int)val;
        } else if (key == "openclip.patch_size" || key == "vision.patch_size") {
            int64_t val; openclip_gguf_read_value(f, type, &val, sizeof(val));
            ctx->hparams.patch_size = (int)val;
        } else {
            openclip_gguf_skip_value(f, type);
        }
    }

    // Patches berechnen
    int grid = ctx->hparams.image_size / ctx->hparams.patch_size;
    ctx->hparams.num_patches = grid * grid;

    OPENCLIP_LOG_INFO("Modell: %s (Hidden: %d, Layers: %d, Heads: %d, Patches: %d)",
                      ctx->model_name.c_str(), ctx->hparams.hidden_size,
                      ctx->hparams.num_hidden_layers, ctx->hparams.num_attention_heads,
                      ctx->hparams.num_patches);

    // Tensoren laden
    if (!openclip_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    OPENCLIP_LOG_INFO("OpenCLIP Modell erfolgreich geladen");
    return ctx;
}

// ============================================================================
// Cleanup
// ============================================================================

void openclip_free(openclip_ctx * ctx) {
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

int openclip_get_dim(const openclip_ctx * ctx) {
    return ctx ? ctx->hparams.hidden_size : 0;
}

int openclip_get_image_size(const openclip_ctx * ctx) {
    return ctx ? ctx->hparams.image_size : 0;
}

const openclip_hparams * openclip_get_hparams(const openclip_ctx * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ============================================================================
// Fehler-API
// ============================================================================

const char * openclip_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}

void openclip_clear_error(void) {
    g_last_error[0] = '\0';
}

void openclip_set_log_level(openclip_log_level level) {
    g_openclip_log_level = level;
}

// ============================================================================
// CGO Kompatibilitaets-Wrapper
// ============================================================================

openclip_init_params openclip_default_params(void) {
    openclip_init_params params = {};
    params.n_threads = 0;  // Auto
    params.n_gpu_layers = -1;  // Alle
    params.main_gpu = 0;
    params.use_mmap = 1;
    params.use_mlock = 0;
    return params;
}

openclip_ctx * openclip_init(const char * model_path, openclip_init_params params) {
    int n_threads = (params.n_threads > 0) ? params.n_threads : 0;
    return openclip_load(model_path, n_threads);
}

openclip_model_info openclip_get_model_info(openclip_ctx * ctx) {
    openclip_model_info info = {};
    if (ctx) {
        info.name = ctx->model_name.c_str();
        info.embedding_dim = ctx->hparams.hidden_size;
        info.image_size = ctx->hparams.image_size;
    }
    return info;
}

int32_t openclip_encode_image(
    openclip_ctx * ctx,
    const uint8_t * image_data,
    size_t image_size,
    float * embedding_out,
    int32_t embedding_dim
) {
    int result = openclip_encode(ctx, image_data, image_size, embedding_out, embedding_dim);
    return (result > 0) ? OPENCLIP_SUCCESS : result;
}

int32_t openclip_encode_batch(
    openclip_ctx * ctx,
    const uint8_t ** images,
    const size_t * sizes,
    int32_t batch_size,
    float * embeddings_out,
    int32_t embedding_dim
) {
    if (!ctx || !images || !sizes || !embeddings_out) {
        return OPENCLIP_ERR_NULL_CTX;
    }

    // Jedes Bild einzeln encodieren
    for (int32_t i = 0; i < batch_size; i++) {
        float * out_ptr = embeddings_out + (i * embedding_dim);
        int result = openclip_encode(ctx, images[i], sizes[i], out_ptr, embedding_dim);
        if (result < 0) {
            return OPENCLIP_ERR_ENCODE;
        }
    }

    return OPENCLIP_SUCCESS;
}
