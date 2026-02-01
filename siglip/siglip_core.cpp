/**
 * siglip_core.cpp - SigLIP Kontext-Verwaltung, Model Loading, GGUF Parsing
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - Interne Strukturen (siglip_ctx)
 * - Logging und Fehlerbehandlung
 * - GGUF-Datei Parsing
 * - Modell-Laden und Tensor-Zuweisung
 * - Modell-Info Abfragen
 */

#include "siglip.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <thread>

// GGML Headers (aus llama.cpp)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

// ============================================================================
// Konstanten
// ============================================================================

// GGUF Magic und Version
constexpr uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

// Maximale String-Laenge fuer Fehler
constexpr size_t MAX_ERROR_LEN = 512;

// ============================================================================
// Globale Variablen (Thread-lokal wo noetig)
// ============================================================================

// Thread-lokaler Fehler-String
static thread_local char g_last_error[MAX_ERROR_LEN] = {0};
static siglip_log_level g_log_level = SIGLIP_LOG_INFO;
static siglip_log_callback g_log_callback = nullptr;
static void * g_log_user_data = nullptr;

// ============================================================================
// Interne Kontext-Struktur
// ============================================================================

/**
 * Interner Kontext - haelt alle Modell-Daten und GGML Ressourcen
 */
struct siglip_ctx {
    // Modell-Info
    std::string model_path;
    std::string model_name;
    siglip_hparams hparams;
    siglip_params params;

    // GGML Kontext
    ggml_context * ctx_data = nullptr;      // Tensor-Daten
    ggml_context * ctx_compute = nullptr;   // Compute-Graph
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Tensor-Referenzen
    struct {
        // Patch Embedding
        ggml_tensor * patch_embed_weight = nullptr;  // [hidden, channels, patch, patch]
        ggml_tensor * patch_embed_bias = nullptr;    // [hidden]
        ggml_tensor * pos_embed = nullptr;           // [num_patches, hidden]

        // Transformer Blocks
        struct block {
            // Attention
            ggml_tensor * attn_q_weight = nullptr;
            ggml_tensor * attn_q_bias = nullptr;
            ggml_tensor * attn_k_weight = nullptr;
            ggml_tensor * attn_k_bias = nullptr;
            ggml_tensor * attn_v_weight = nullptr;
            ggml_tensor * attn_v_bias = nullptr;
            ggml_tensor * attn_out_weight = nullptr;
            ggml_tensor * attn_out_bias = nullptr;

            // MLP
            ggml_tensor * mlp_fc1_weight = nullptr;
            ggml_tensor * mlp_fc1_bias = nullptr;
            ggml_tensor * mlp_fc2_weight = nullptr;
            ggml_tensor * mlp_fc2_bias = nullptr;

            // LayerNorm
            ggml_tensor * ln1_weight = nullptr;
            ggml_tensor * ln1_bias = nullptr;
            ggml_tensor * ln2_weight = nullptr;
            ggml_tensor * ln2_bias = nullptr;
        };
        std::vector<block> blocks;

        // Output
        ggml_tensor * norm_weight = nullptr;
        ggml_tensor * norm_bias = nullptr;
        ggml_tensor * head_weight = nullptr;  // Optional projection
        ggml_tensor * head_bias = nullptr;
    } tensors;
};

// ============================================================================
// Hilfsfunktionen - Fehler und Logging
// ============================================================================

/**
 * Setzt den letzten Fehler-String (Thread-sicher)
 */
static void set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

/**
 * Loggt eine Nachricht mit Level-Filter
 */
static void log_msg(siglip_log_level level, const char * fmt, ...) {
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

// Log-Makros fuer bequeme Nutzung
#define LOG_ERROR(...) log_msg(SIGLIP_LOG_ERROR, __VA_ARGS__)
#define LOG_WARN(...)  log_msg(SIGLIP_LOG_WARN, __VA_ARGS__)
#define LOG_INFO(...)  log_msg(SIGLIP_LOG_INFO, __VA_ARGS__)
#define LOG_DEBUG(...) log_msg(SIGLIP_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Parsing - Liest GGUF-Metadaten und Tensoren
// ============================================================================

// GGUF Metadaten-Typen
enum gguf_type {
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

/**
 * Liest einen String aus GGUF-Format (uint64 Laenge + Daten)
 */
static std::string read_gguf_string(FILE * f) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return "";
    std::string s(len, '\0');
    if (fread(&s[0], 1, len, f) != len) return "";
    return s;
}

/**
 * Liest einen Metadaten-Wert basierend auf Typ
 */
static bool read_gguf_metadata_value(FILE * f, uint32_t type, void * out, size_t max_size) {
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
            *str = read_gguf_string(f);
            return !str->empty() || feof(f) == 0;
        }
        default:
            return false;
    }
}

// ============================================================================
// Modell laden - Tensor-Parsing und Zuweisung
// ============================================================================

/**
 * Laedt alle Tensoren aus der GGUF-Datei und weist sie dem Kontext zu
 */
static bool load_model_tensors(siglip_ctx * ctx, FILE * f, uint64_t n_tensors) {
    LOG_DEBUG("Lade %lu Tensoren...", (unsigned long)n_tensors);

    // Tensor-Info Struktur fuer temporaere Speicherung
    struct tensor_info {
        std::string name;
        uint32_t n_dims;
        std::vector<uint64_t> dims;
        uint32_t type;
        uint64_t offset;
    };
    std::vector<tensor_info> tensor_infos(n_tensors);

    // Tensor-Infos aus Header lesen
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_infos[i].name = read_gguf_string(f);
        fread(&tensor_infos[i].n_dims, sizeof(uint32_t), 1, f);

        tensor_infos[i].dims.resize(tensor_infos[i].n_dims);
        for (uint32_t j = 0; j < tensor_infos[i].n_dims; j++) {
            fread(&tensor_infos[i].dims[j], sizeof(uint64_t), 1, f);
        }

        fread(&tensor_infos[i].type, sizeof(uint32_t), 1, f);
        fread(&tensor_infos[i].offset, sizeof(uint64_t), 1, f);

        LOG_DEBUG("  Tensor %lu: %s [%u dims], type=%u", i, tensor_infos[i].name.c_str(),
                  tensor_infos[i].n_dims, tensor_infos[i].type);
    }

    // Berechne Gesamtgroesse aller Tensoren
    size_t total_size = 0;
    for (const auto & ti : tensor_infos) {
        size_t n_elements = 1;
        for (auto d : ti.dims) n_elements *= d;

        size_t element_size = 4; // Default F32
        if (ti.type == GGML_TYPE_F16) element_size = 2;
        else if (ti.type == GGML_TYPE_Q8_0) element_size = 1; // Approximation

        total_size += n_elements * element_size;
    }

    // GGML Kontext fuer Tensor-Daten erstellen
    ggml_init_params ggml_params = {
        .mem_size   = total_size + 256 * 1024 * 1024, // Extra fuer Overhead
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ctx->ctx_data = ggml_init(ggml_params);
    if (!ctx->ctx_data) {
        set_error("Konnte GGML Kontext nicht erstellen");
        return false;
    }

    // Alignment auf 32 bytes fuer Daten-Start
    long current_pos = ftell(f);
    long alignment = 32;
    long padding = (alignment - (current_pos % alignment)) % alignment;
    fseek(f, padding, SEEK_CUR);
    long data_start = ftell(f);

    // Block-Array initialisieren
    ctx->tensors.blocks.resize(ctx->hparams.num_hidden_layers);

    // Tensoren erstellen und Daten laden
    for (const auto & ti : tensor_infos) {
        // GGML Tensor erstellen mit korrekten Dimensionen
        std::vector<int64_t> ne(4, 1);
        for (uint32_t j = 0; j < ti.n_dims && j < 4; j++) {
            ne[j] = ti.dims[j];
        }

        ggml_tensor * tensor = ggml_new_tensor_4d(
            ctx->ctx_data,
            static_cast<ggml_type>(ti.type),
            ne[0], ne[1], ne[2], ne[3]
        );
        ggml_set_name(tensor, ti.name.c_str());

        // Tensor-Daten aus Datei laden
        fseek(f, data_start + ti.offset, SEEK_SET);
        size_t bytes = ggml_nbytes(tensor);
        if (fread(tensor->data, 1, bytes, f) != bytes) {
            LOG_ERROR("Fehler beim Laden von Tensor %s", ti.name.c_str());
            return false;
        }

        // Tensor der richtigen Komponente zuweisen
        const std::string & name = ti.name;

        // Patch Embedding Tensoren
        if (name == "siglip.patch_embed.weight") {
            ctx->tensors.patch_embed_weight = tensor;
        } else if (name == "siglip.patch_embed.bias") {
            ctx->tensors.patch_embed_bias = tensor;
        } else if (name == "siglip.pos_embed") {
            ctx->tensors.pos_embed = tensor;
        }
        // Output Tensoren
        else if (name == "siglip.norm.weight") {
            ctx->tensors.norm_weight = tensor;
        } else if (name == "siglip.norm.bias") {
            ctx->tensors.norm_bias = tensor;
        } else if (name == "siglip.head.weight") {
            ctx->tensors.head_weight = tensor;
        } else if (name == "siglip.head.bias") {
            ctx->tensors.head_bias = tensor;
        }
        // Transformer Block Tensoren
        else if (name.find("siglip.blocks.") == 0) {
            // Block-Tensor parsen: siglip.blocks.{i}.{component}
            size_t dot1 = name.find('.', 14); // Nach "siglip.blocks."
            if (dot1 != std::string::npos) {
                int block_idx = std::stoi(name.substr(14, dot1 - 14));
                std::string component = name.substr(dot1 + 1);

                if (block_idx >= 0 && block_idx < ctx->hparams.num_hidden_layers) {
                    auto & block = ctx->tensors.blocks[block_idx];

                    // Attention Tensoren
                    if (component == "attn.q.weight") block.attn_q_weight = tensor;
                    else if (component == "attn.q.bias") block.attn_q_bias = tensor;
                    else if (component == "attn.k.weight") block.attn_k_weight = tensor;
                    else if (component == "attn.k.bias") block.attn_k_bias = tensor;
                    else if (component == "attn.v.weight") block.attn_v_weight = tensor;
                    else if (component == "attn.v.bias") block.attn_v_bias = tensor;
                    else if (component == "attn.out.weight") block.attn_out_weight = tensor;
                    else if (component == "attn.out.bias") block.attn_out_bias = tensor;
                    // MLP Tensoren
                    else if (component == "mlp.fc1.weight") block.mlp_fc1_weight = tensor;
                    else if (component == "mlp.fc1.bias") block.mlp_fc1_bias = tensor;
                    else if (component == "mlp.fc2.weight") block.mlp_fc2_weight = tensor;
                    else if (component == "mlp.fc2.bias") block.mlp_fc2_bias = tensor;
                    // LayerNorm Tensoren
                    else if (component == "ln1.weight") block.ln1_weight = tensor;
                    else if (component == "ln1.bias") block.ln1_bias = tensor;
                    else if (component == "ln2.weight") block.ln2_weight = tensor;
                    else if (component == "ln2.bias") block.ln2_bias = tensor;
                }
            }
        }
    }

    LOG_INFO("Tensoren geladen: %lu", (unsigned long)n_tensors);
    return true;
}

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
    const char * model_path,
    siglip_params params,
    siglip_progress_callback callback,
    void * user_data
) {
    LOG_INFO("Lade Modell: %s", model_path);

    FILE * f = fopen(model_path, "rb");
    if (!f) {
        set_error("Konnte Datei nicht oeffnen: %s", model_path);
        return nullptr;
    }

    // Magic pruefen
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GGUF_MAGIC) {
        set_error("Ungueltige GGUF Magic: 0x%08X (erwartet 0x%08X)", magic, GGUF_MAGIC);
        fclose(f);
        return nullptr;
    }

    // Version pruefen
    uint32_t version;
    fread(&version, sizeof(version), 1, f);
    if (version < 2 || version > 3) {
        LOG_WARN("Unbekannte GGUF Version: %u", version);
    }

    // Header lesen
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, sizeof(n_tensors), 1, f);
    fread(&n_kv, sizeof(n_kv), 1, f);

    LOG_DEBUG("GGUF v%u: %lu Tensoren, %lu Metadaten", version, n_tensors, n_kv);

    // Kontext erstellen
    siglip_ctx * ctx = new siglip_ctx();
    ctx->model_path = model_path;
    ctx->params = params;

    // Default Hyperparameter (werden durch Metadaten ueberschrieben)
    ctx->hparams.model_type = SIGLIP_MODEL_VIT_B_16;
    ctx->hparams.hidden_size = 768;
    ctx->hparams.intermediate_size = 3072;
    ctx->hparams.num_attention_heads = 12;
    ctx->hparams.num_hidden_layers = 12;
    ctx->hparams.image_size = 224;
    ctx->hparams.patch_size = 16;
    ctx->hparams.num_patches = 196; // (224/16)^2
    ctx->hparams.layer_norm_eps = 1e-6f;

    // Default Preprocessing Parameter
    ctx->hparams.preprocess.target_size = 224;
    ctx->hparams.preprocess.mean[0] = 0.5f;
    ctx->hparams.preprocess.mean[1] = 0.5f;
    ctx->hparams.preprocess.mean[2] = 0.5f;
    ctx->hparams.preprocess.std[0] = 0.5f;
    ctx->hparams.preprocess.std[1] = 0.5f;
    ctx->hparams.preprocess.std[2] = 0.5f;
    ctx->hparams.preprocess.center_crop = false;
    ctx->hparams.preprocess.bicubic = true;

    // Metadaten aus GGUF lesen
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_gguf_string(f);
        uint32_t type;
        fread(&type, sizeof(type), 1, f);

        if (key == "general.architecture") {
            std::string arch;
            read_gguf_metadata_value(f, type, &arch, 0);
            if (arch != "siglip") {
                LOG_WARN("Unerwartete Architektur: %s", arch.c_str());
            }
        } else if (key == "general.name") {
            read_gguf_metadata_value(f, type, &ctx->model_name, 0);
        } else if (key == "siglip.hidden_size") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.hidden_size = static_cast<int>(val);
        } else if (key == "siglip.num_attention_heads") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.num_attention_heads = static_cast<int>(val);
        } else if (key == "siglip.num_hidden_layers") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.num_hidden_layers = static_cast<int>(val);
        } else if (key == "siglip.intermediate_size") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.intermediate_size = static_cast<int>(val);
        } else if (key == "siglip.image_size") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.image_size = static_cast<int>(val);
            ctx->hparams.preprocess.target_size = static_cast<int>(val);
        } else if (key == "siglip.patch_size") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.patch_size = static_cast<int>(val);
        } else if (key == "siglip.num_patches") {
            int64_t val;
            read_gguf_metadata_value(f, type, &val, sizeof(val));
            ctx->hparams.num_patches = static_cast<int>(val);
        } else if (key == "siglip.image_mean") {
            // Array lesen
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, sizeof(arr_type), 1, f);
            fread(&arr_len, sizeof(arr_len), 1, f);
            for (uint64_t j = 0; j < arr_len && j < 3; j++) {
                float val;
                fread(&val, sizeof(val), 1, f);
                ctx->hparams.preprocess.mean[j] = val;
            }
        } else if (key == "siglip.image_std") {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, sizeof(arr_type), 1, f);
            fread(&arr_len, sizeof(arr_len), 1, f);
            for (uint64_t j = 0; j < arr_len && j < 3; j++) {
                float val;
                fread(&val, sizeof(val), 1, f);
                ctx->hparams.preprocess.std[j] = val;
            }
        } else {
            // Unbekannte Metadaten ueberspringen
            if (type == GGUF_TYPE_STRING) {
                read_gguf_string(f);
            } else if (type <= GGUF_TYPE_FLOAT64 && type != GGUF_TYPE_STRING) {
                char buf[8];
                size_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
                if (type < sizeof(sizes)/sizeof(sizes[0])) {
                    fread(buf, sizes[type], 1, f);
                }
            }
        }

        // Progress Callback aufrufen
        if (callback) {
            callback(static_cast<float>(i + 1) / n_kv * 0.5f, user_data);
        }
    }

    // Modell-Typ basierend auf hidden_size bestimmen
    if (ctx->hparams.hidden_size <= 768) {
        ctx->hparams.model_type = SIGLIP_MODEL_VIT_B_16;
    } else if (ctx->hparams.hidden_size <= 1024) {
        ctx->hparams.model_type = SIGLIP_MODEL_VIT_L_16;
    } else {
        ctx->hparams.model_type = SIGLIP_MODEL_VIT_SO400M;
    }

    LOG_INFO("Modell: %s", ctx->model_name.c_str());
    LOG_INFO("  Hidden Size: %d", ctx->hparams.hidden_size);
    LOG_INFO("  Layers: %d", ctx->hparams.num_hidden_layers);
    LOG_INFO("  Heads: %d", ctx->hparams.num_attention_heads);
    LOG_INFO("  Image Size: %d", ctx->hparams.image_size);
    LOG_INFO("  Patch Size: %d", ctx->hparams.patch_size);

    // Tensoren laden
    if (!load_model_tensors(ctx, f, n_tensors)) {
        delete ctx;
        fclose(f);
        return nullptr;
    }

    fclose(f);

    if (callback) {
        callback(1.0f, user_data);
    }

    LOG_INFO("Modell geladen");
    return ctx;
}

void siglip_free(siglip_ctx * ctx) {
    if (!ctx) return;

    // GGML Ressourcen freigeben
    if (ctx->allocr) {
        ggml_gallocr_free(ctx->allocr);
    }
    if (ctx->buffer) {
        ggml_backend_buffer_free(ctx->buffer);
    }
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    if (ctx->ctx_compute) {
        ggml_free(ctx->ctx_compute);
    }
    if (ctx->ctx_data) {
        ggml_free(ctx->ctx_data);
    }

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

void siglip_set_log_level(siglip_log_level level) {
    g_log_level = level;
}

void siglip_set_log_callback(siglip_log_callback callback, void * user_data) {
    g_log_callback = callback;
    g_log_user_data = user_data;
}

// ============================================================================
// Oeffentliche API - System-Info
// ============================================================================

const char * siglip_version(void) {
    return "0.1.0";
}

const char * siglip_build_info(void) {
    static std::string info;
    if (info.empty()) {
        info = "siglip built with:";

        #ifdef __GNUC__
        info += " GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
        #endif

        #ifdef _MSC_VER
        info += " MSVC " + std::to_string(_MSC_VER);
        #endif

        #ifdef GGML_USE_CUDA
        info += " CUDA";
        #endif

        #ifdef GGML_USE_METAL
        info += " Metal";
        #endif

        #ifdef __AVX2__
        info += " AVX2";
        #endif

        #ifdef __AVX512F__
        info += " AVX512";
        #endif
    }
    return info.c_str();
}

bool siglip_backend_available(siglip_backend backend) {
    switch (backend) {
        case SIGLIP_BACKEND_CPU:
            return true;
        case SIGLIP_BACKEND_CUDA:
            #ifdef GGML_USE_CUDA
            return true;
            #else
            return false;
            #endif
        case SIGLIP_BACKEND_METAL:
            #ifdef GGML_USE_METAL
            return true;
            #else
            return false;
            #endif
        case SIGLIP_BACKEND_VULKAN:
            return false; // Noch nicht implementiert
        default:
            return false;
    }
}

int siglip_get_available_backends(siglip_backend * backends, int max_backends) {
    int n = 0;

    if (n < max_backends) backends[n++] = SIGLIP_BACKEND_CPU;

    #ifdef GGML_USE_CUDA
    if (n < max_backends) backends[n++] = SIGLIP_BACKEND_CUDA;
    #endif

    #ifdef GGML_USE_METAL
    if (n < max_backends) backends[n++] = SIGLIP_BACKEND_METAL;
    #endif

    return n;
}

const char * siglip_system_info(void) {
    return ggml_cpu_has_avx() ? "AVX " : ""
         + std::string(ggml_cpu_has_avx2() ? "AVX2 " : "")
         + std::string(ggml_cpu_has_avx512() ? "AVX512 " : "")
         + std::string(ggml_cpu_has_fma() ? "FMA " : "");
}
