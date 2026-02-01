/**
 * siglip_core.cpp - SigLIP Kontext-Verwaltung und Model Loading
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - Globale Variablen (Log-Level, Callbacks)
 * - Logging und Fehlerbehandlung
 * - Modell-Laden aus GGUF
 * - Modell-Info Abfragen
 */

#include "siglip_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[SIGLIP_MAX_ERROR_LEN] = {0};

// Log-Variablen (NICHT static - werden von siglip_system.cpp extern referenziert)
siglip_log_level g_log_level = SIGLIP_LOG_INFO;
siglip_log_callback g_log_callback = nullptr;
void * g_log_user_data = nullptr;

// ============================================================================
// Interne Hilfsfunktionen - Fehler und Logging
// ============================================================================

void siglip_set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, SIGLIP_MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

void siglip_log_msg(siglip_log_level level, const char * fmt, ...) {
    if (level > g_log_level) return;

    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    if (g_log_callback) {
        g_log_callback(level, buffer, g_log_user_data);
    } else {
        const char * prefix = "";
        switch (level) {
            case SIGLIP_LOG_ERROR: prefix = "[ERROR] "; break;
            case SIGLIP_LOG_WARN:  prefix = "[WARN]  "; break;
            case SIGLIP_LOG_INFO:  prefix = "[INFO]  "; break;
            case SIGLIP_LOG_DEBUG: prefix = "[DEBUG] "; break;
            default: break;
        }
        fprintf(stderr, "siglip: %s%s\n", prefix, buffer);
    }
}

// ============================================================================
// GGUF Konstanten (lokal)
// ============================================================================

constexpr uint32_t GGUF_TYPE_STRING  = 8;
constexpr uint32_t GGUF_TYPE_FLOAT64 = 12;

// ============================================================================
// Oeffentliche API - Modell-Verwaltung
// ============================================================================

siglip_params siglip_params_default(void) {
    siglip_params params = {};
    params.backend = SIGLIP_BACKEND_CPU;
    params.log_level = SIGLIP_LOG_INFO;
    params.embed_format = SIGLIP_EMBED_F32;
    params.n_threads = std::thread::hardware_concurrency();
    params.n_gpu_layers = -1;
    params.main_gpu = 0;
    params.use_mmap = true;
    params.use_mlock = false;
    params.batch_size = 1;
    return params;
}

siglip_ctx * siglip_load_model(const char * model_path, siglip_params params) {
    return siglip_load_model_with_progress(model_path, params, nullptr, nullptr);
}

siglip_ctx * siglip_load_model_with_progress(
    const char * model_path, siglip_params params,
    siglip_progress_callback callback, void * user_data
) {
    SIGLIP_LOG_INFO("Lade Modell: %s", model_path);

    FILE * f = fopen(model_path, "rb");
    if (!f) {
        siglip_set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // Magic und Version pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != SIGLIP_GGUF_MAGIC) {
        siglip_set_error("Ungueltige GGUF Magic: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    if (version < 2 || version > 3) {
        SIGLIP_LOG_WARN("Unbekannte GGUF Version: %u", version);
    }

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    SIGLIP_LOG_DEBUG("GGUF v%u: %lu Tensoren, %lu Metadaten", version, n_tensors, n_kv);

    // Kontext erstellen mit Default-Werten
    siglip_ctx * ctx = new siglip_ctx();
    ctx->model_path = model_path;
    ctx->params = params;
    ctx->hparams.model_type = SIGLIP_MODEL_VIT_B_16;
    ctx->hparams.hidden_size = 768;
    ctx->hparams.intermediate_size = 3072;
    ctx->hparams.num_attention_heads = 12;
    ctx->hparams.num_hidden_layers = 12;
    ctx->hparams.image_size = 224;
    ctx->hparams.patch_size = 16;
    ctx->hparams.num_patches = 196;
    ctx->hparams.layer_norm_eps = 1e-6f;
    ctx->hparams.preprocess.target_size = 224;
    for (int i = 0; i < 3; i++) {
        ctx->hparams.preprocess.mean[i] = 0.5f;
        ctx->hparams.preprocess.std[i] = 0.5f;
    }
    ctx->hparams.preprocess.center_crop = false;
    ctx->hparams.preprocess.bicubic = true;

    // Metadaten lesen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = gguf_read_string(f);
        uint32_t type;
        fread(&type, 4, 1, f);

        if (key == "general.name") {
            gguf_read_metadata_value(f, type, &ctx->model_name, 0);
        } else if (key == "siglip.hidden_size") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "siglip.num_attention_heads") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "siglip.num_hidden_layers") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "siglip.intermediate_size") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.intermediate_size = (int)val;
        } else if (key == "siglip.image_size") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.image_size = (int)val;
            ctx->hparams.preprocess.target_size = (int)val;
        } else if (key == "siglip.patch_size") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.patch_size = (int)val;
        } else if (key == "siglip.num_patches") {
            int64_t val; gguf_read_metadata_value(f, type, &val, 8);
            ctx->hparams.num_patches = (int)val;
        } else {
            gguf_skip_metadata_value(f, type);
        }
        if (callback) callback((float)(i + 1) / n_kv * 0.5f, user_data);
    }

    // Modell-Typ bestimmen
    if (ctx->hparams.hidden_size <= 768) ctx->hparams.model_type = SIGLIP_MODEL_VIT_B_16;
    else if (ctx->hparams.hidden_size <= 1024) ctx->hparams.model_type = SIGLIP_MODEL_VIT_L_16;
    else ctx->hparams.model_type = SIGLIP_MODEL_VIT_SO400M;

    SIGLIP_LOG_INFO("Modell: %s (Hidden: %d, Layers: %d)", ctx->model_name.c_str(),
                    ctx->hparams.hidden_size, ctx->hparams.num_hidden_layers);

    // Tensoren laden (aus siglip_gguf.cpp)
    if (!siglip_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    if (callback) callback(1.0f, user_data);
    SIGLIP_LOG_INFO("Modell geladen");
    return ctx;
}

void siglip_free(siglip_ctx * ctx) {
    if (!ctx) return;
    if (ctx->allocr) ggml_gallocr_free(ctx->allocr);
    if (ctx->buffer) ggml_backend_buffer_free(ctx->buffer);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    if (ctx->ctx_compute) ggml_free(ctx->ctx_compute);
    if (ctx->ctx_data) ggml_free(ctx->ctx_data);
    delete ctx;
}

// ============================================================================
// Oeffentliche API - Modell-Info Abfragen
// ============================================================================

const siglip_hparams * siglip_get_hparams(const siglip_ctx * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

int siglip_get_embedding_dim(const siglip_ctx * ctx) {
    return ctx ? ctx->hparams.hidden_size : 0;
}

int siglip_get_image_size(const siglip_ctx * ctx) {
    return ctx ? ctx->hparams.image_size : 0;
}

siglip_model_type siglip_get_model_type(const siglip_ctx * ctx) {
    return ctx ? ctx->hparams.model_type : SIGLIP_MODEL_UNKNOWN;
}

const char * siglip_get_model_name(const siglip_ctx * ctx) {
    return ctx ? ctx->model_name.c_str() : nullptr;
}

// ============================================================================
// Oeffentliche API - Fehlerbehandlung
// ============================================================================

const char * siglip_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}

void siglip_clear_error(void) {
    g_last_error[0] = '\0';
}
