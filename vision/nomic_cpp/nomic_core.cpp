/**
 * MODUL: nomic_core.cpp
 * ZWECK: Nomic Vision Kontext-Verwaltung, GGUF Parsing, Tensor-Loading
 * INPUT: GGUF-Datei mit nomic-embed-vision Gewichten
 * OUTPUT: nomic_ctx mit initialisierten Tensoren
 * NEBENEFFEKTE: Speicherallokation, Datei-I/O, Logging
 * ABHAENGIGKEITEN: nomic_internal.h, ggml
 * HINWEISE: GGUF v3 Format, Tensor-Namen folgen v.* Konvention
 */

#include "nomic_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// Globale Variablen
// ============================================================================

static thread_local char g_last_error[NOMIC_MAX_ERROR_LEN] = {0};
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
    if (level > g_nomic_log_level) return;

    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

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
    if (len == 0) return "";

    std::string result(len, '\0');
    if (fread(&result[0], 1, len, f) != len) return "";
    return result;
}

bool nomic_gguf_read_value(FILE * f, uint32_t type, void * out, size_t max_size) {
    switch (type) {
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: {
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) return false;
            if (out && max_size >= 4) *(uint32_t*)out = val;
            return true;
        }
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: {
            uint64_t val;
            if (fread(&val, 8, 1, f) != 1) return false;
            if (out && max_size >= 8) *(uint64_t*)out = val;
            return true;
        }
        case GGUF_TYPE_STRING: {
            std::string str = nomic_gguf_read_string(f);
            if (out) *(std::string*)out = str;
            return true;
        }
        default:
            return nomic_gguf_skip_value(f, type);
    }
}

bool nomic_gguf_skip_value(FILE * f, uint32_t type) {
    static const size_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

    if (type == GGUF_TYPE_STRING) {
        nomic_gguf_read_string(f);
        return true;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t arr_type; uint64_t arr_len;
        fread(&arr_type, 4, 1, f);
        fread(&arr_len, 8, 1, f);
        for (uint64_t i = 0; i < arr_len; i++) {
            nomic_gguf_skip_value(f, arr_type);
        }
        return true;
    }
    if (type < 13 && sizes[type] > 0) {
        return fseek(f, (long)sizes[type], SEEK_CUR) == 0;
    }
    return false;
}

// ============================================================================
// Tensor-Info fuer Parsing
// ============================================================================

struct tensor_info {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
};

// ============================================================================
// Tensor-Loading
// ============================================================================

bool nomic_load_tensors(nomic_ctx * ctx, FILE * f, uint64_t n_tensors) {
    NOMIC_LOG_DBG("Lade %lu Tensoren...", (unsigned long)n_tensors);

    // Tensor-Infos aus GGUF Header lesen
    std::vector<tensor_info> infos(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        infos[i].name = nomic_gguf_read_string(f);
        fread(&infos[i].n_dims, 4, 1, f);

        infos[i].dims.resize(infos[i].n_dims);
        for (uint32_t d = 0; d < infos[i].n_dims; d++) {
            fread(&infos[i].dims[d], 8, 1, f);
        }
        fread(&infos[i].type, 4, 1, f);
        fread(&infos[i].offset, 8, 1, f);

        NOMIC_LOG_DBG("  [%lu] %s dims=%u type=%u", i, infos[i].name.c_str(),
                      infos[i].n_dims, infos[i].type);
    }

    // GGML Kontext erstellen (256MB fuer Gewichte)
    size_t mem_size = 256 * 1024 * 1024;
    ggml_init_params params = {mem_size, nullptr, false};
    ctx->ctx_data = ggml_init(params);
    if (!ctx->ctx_data) {
        nomic_set_error("GGML Kontext konnte nicht erstellt werden");
        return false;
    }

    // Backend initialisieren (CPU)
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        nomic_set_error("CPU Backend konnte nicht initialisiert werden");
        return false;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Data-Start Position (32-byte aligned)
    long pos = ftell(f);
    long padding = (32 - (pos % 32)) % 32;
    long data_start = pos + padding;

    // Layer-Vektor initialisieren
    ctx->tensors.layers.resize(ctx->hparams.num_hidden_layers);

    // Tensoren erstellen und laden
    for (const auto & ti : infos) {
        // GGML Tensor erstellen
        std::vector<int64_t> ne(4, 1);
        for (uint32_t d = 0; d < ti.n_dims && d < 4; d++) {
            ne[d] = (int64_t)ti.dims[d];
        }

        ggml_tensor * tensor = ggml_new_tensor_4d(
            ctx->ctx_data, (ggml_type)ti.type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(tensor, ti.name.c_str());

        // Daten aus Datei laden
        fseek(f, data_start + (long)ti.offset, SEEK_SET);
        size_t bytes = ggml_nbytes(tensor);
        if (fread(tensor->data, 1, bytes, f) != bytes) {
            NOMIC_LOG_ERR("Fehler beim Laden: %s", ti.name.c_str());
            return false;
        }

        // Tensor zuweisen (v.* Namenskonvention vom Konverter)
        const std::string & name = ti.name;

        // Globale Tensoren
        if (name == "v.patch_emb.weight") ctx->tensors.patch_embed_weight = tensor;
        else if (name == "v.patch_emb.bias") ctx->tensors.patch_embed_bias = tensor;
        else if (name == "v.pos_emb") ctx->tensors.pos_embed = tensor;
        else if (name == "v.cls_token") ctx->tensors.cls_token = tensor;
        else if (name == "v.post_ln.weight") ctx->tensors.post_ln_weight = tensor;
        else if (name == "v.post_ln.bias") ctx->tensors.post_ln_bias = tensor;

        // Block-Tensoren: v.blk.{i}.{component}
        else if (name.find("v.blk.") == 0) {
            size_t dot1 = name.find('.', 6);
            if (dot1 != std::string::npos) {
                int idx = std::stoi(name.substr(6, dot1 - 6));
                std::string comp = name.substr(dot1 + 1);

                if (idx >= 0 && idx < ctx->hparams.num_hidden_layers) {
                    auto & layer = ctx->tensors.layers[idx];

                    // Attention
                    if      (comp == "attn.q.weight") layer.q_weight = tensor;
                    else if (comp == "attn.q.bias")   layer.q_bias = tensor;
                    else if (comp == "attn.k.weight") layer.k_weight = tensor;
                    else if (comp == "attn.k.bias")   layer.k_bias = tensor;
                    else if (comp == "attn.v.weight") layer.v_weight = tensor;
                    else if (comp == "attn.v.bias")   layer.v_bias = tensor;
                    else if (comp == "attn.out.weight") layer.o_weight = tensor;
                    else if (comp == "attn.out.bias")   layer.o_bias = tensor;

                    // MLP (SwiGLU oder Standard)
                    else if (comp == "ffn.gate.weight") layer.ffn_gate_weight = tensor;
                    else if (comp == "ffn.gate.bias")   layer.ffn_gate_bias = tensor;
                    else if (comp == "ffn.up.weight")   layer.ffn_up_weight = tensor;
                    else if (comp == "ffn.up.bias")     layer.ffn_up_bias = tensor;
                    else if (comp == "ffn.down.weight") layer.ffn_down_weight = tensor;
                    else if (comp == "ffn.down.bias")   layer.ffn_down_bias = tensor;

                    // Layer Norm
                    else if (comp == "ln1.weight") layer.ln1_weight = tensor;
                    else if (comp == "ln1.bias")   layer.ln1_bias = tensor;
                    else if (comp == "ln2.weight") layer.ln2_weight = tensor;
                    else if (comp == "ln2.bias")   layer.ln2_bias = tensor;
                }
            }
        }
    }

    NOMIC_LOG_INFO("Tensoren geladen: %lu", (unsigned long)n_tensors);
    return true;
}

// ============================================================================
// Oeffentliche API - Init/Free
// ============================================================================

nomic_ctx * nomic_init(const char * model_path, int n_threads) {
    NOMIC_LOG_INFO("Lade: %s", model_path);

    FILE * f = fopen(model_path, "rb");
    if (!f) {
        nomic_set_error("Datei nicht gefunden: %s", model_path);
        return nullptr;
    }

    // GGUF Magic pruefen
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || magic != NOMIC_GGUF_MAGIC) {
        nomic_set_error("Keine gueltige GGUF-Datei: 0x%08X", magic);
        fclose(f);
        return nullptr;
    }
    fread(&version, 4, 1, f);
    NOMIC_LOG_DBG("GGUF Version: %u", version);

    // Header
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    // Kontext mit Defaults
    nomic_ctx * ctx = new nomic_ctx();
    ctx->model_path = model_path;
    ctx->n_threads = (n_threads > 0) ? n_threads : (int)std::thread::hardware_concurrency();

    // Default Hyperparameter (nomic-embed-vision-v1.5)
    ctx->hparams.hidden_size = NOMIC_DEFAULT_HIDDEN_SIZE;
    ctx->hparams.intermediate_size = 3072;
    ctx->hparams.num_attention_heads = NOMIC_MAX_HEADS;
    ctx->hparams.num_hidden_layers = NOMIC_MAX_LAYERS;
    ctx->hparams.image_size = NOMIC_DEFAULT_IMAGE_SIZE;
    ctx->hparams.patch_size = NOMIC_DEFAULT_PATCH_SIZE;
    ctx->hparams.num_patches = (384 / 14) * (384 / 14);  // 729
    ctx->hparams.layer_norm_eps = 1e-6f;

    // Metadaten parsen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = nomic_gguf_read_string(f);
        uint32_t type; fread(&type, 4, 1, f);

        if (key == "general.name") {
            nomic_gguf_read_value(f, type, &ctx->model_name, 0);
        } else if (key == "nomic.hidden_size") {
            uint32_t val; nomic_gguf_read_value(f, type, &val, 4);
            ctx->hparams.hidden_size = (int)val;
        } else if (key == "nomic.num_attention_heads") {
            uint32_t val; nomic_gguf_read_value(f, type, &val, 4);
            ctx->hparams.num_attention_heads = (int)val;
        } else if (key == "nomic.num_hidden_layers") {
            uint32_t val; nomic_gguf_read_value(f, type, &val, 4);
            ctx->hparams.num_hidden_layers = (int)val;
        } else if (key == "nomic.image_size") {
            uint32_t val; nomic_gguf_read_value(f, type, &val, 4);
            ctx->hparams.image_size = (int)val;
        } else if (key == "nomic.patch_size") {
            uint32_t val; nomic_gguf_read_value(f, type, &val, 4);
            ctx->hparams.patch_size = (int)val;
        } else {
            nomic_gguf_skip_value(f, type);
        }
    }

    // Patches berechnen
    int grid = ctx->hparams.image_size / ctx->hparams.patch_size;
    ctx->hparams.num_patches = grid * grid;

    NOMIC_LOG_INFO("Modell: %s (H=%d, L=%d, P=%d)",
                   ctx->model_name.c_str(), ctx->hparams.hidden_size,
                   ctx->hparams.num_hidden_layers, ctx->hparams.num_patches);

    // Tensoren laden
    if (!nomic_load_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);
    NOMIC_LOG_INFO("Modell bereit");
    return ctx;
}

void nomic_free(nomic_ctx * ctx) {
    if (!ctx) return;
    if (ctx->allocr)      ggml_gallocr_free(ctx->allocr);
    if (ctx->buffer)      ggml_backend_buffer_free(ctx->buffer);
    if (ctx->backend)     ggml_backend_free(ctx->backend);
    if (ctx->ctx_compute) ggml_free(ctx->ctx_compute);
    if (ctx->ctx_data)    ggml_free(ctx->ctx_data);
    delete ctx;
}

// ============================================================================
// Oeffentliche API - Info
// ============================================================================

int nomic_get_embedding_dim(const nomic_ctx * ctx) {
    return ctx ? ctx->hparams.hidden_size : 0;
}

int nomic_get_image_size(const nomic_ctx * ctx) {
    return ctx ? ctx->hparams.image_size : 0;
}

int nomic_get_patch_size(const nomic_ctx * ctx) {
    return ctx ? ctx->hparams.patch_size : 0;
}

const nomic_hparams * nomic_get_hparams(const nomic_ctx * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ============================================================================
// Oeffentliche API - Fehler
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
