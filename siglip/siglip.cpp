/**
 * siglip.cpp - SigLIP Vision Encoder Implementation
 *
 * Implementiert den SigLIP Vision Transformer für llama.cpp.
 * Nutzt GGML für Tensor-Operationen und unterstützt CPU/CUDA/Metal.
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

// STB für Bild-Loading
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

// ============================================================================
// Interne Strukturen
// ============================================================================

// GGUF Magic und Version
constexpr uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

// Maximale String-Länge für Fehler
constexpr size_t MAX_ERROR_LEN = 512;

// Thread-lokaler Fehler-String
static thread_local char g_last_error[MAX_ERROR_LEN] = {0};
static siglip_log_level g_log_level = SIGLIP_LOG_INFO;
static siglip_log_callback g_log_callback = nullptr;
static void * g_log_user_data = nullptr;

/**
 * Interner Kontext
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
// Hilfsfunktionen
// ============================================================================

static void set_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, MAX_ERROR_LEN, fmt, args);
    va_end(args);
}

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

#define LOG_ERROR(...) log_msg(SIGLIP_LOG_ERROR, __VA_ARGS__)
#define LOG_WARN(...)  log_msg(SIGLIP_LOG_WARN, __VA_ARGS__)
#define LOG_INFO(...)  log_msg(SIGLIP_LOG_INFO, __VA_ARGS__)
#define LOG_DEBUG(...) log_msg(SIGLIP_LOG_DEBUG, __VA_ARGS__)

// ============================================================================
// GGUF Parsing
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

// Liest String aus GGUF
static std::string read_gguf_string(FILE * f) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return "";
    std::string s(len, '\0');
    if (fread(&s[0], 1, len, f) != len) return "";
    return s;
}

// Liest Metadaten-Wert
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
// Modell laden
// ============================================================================

static bool load_model_tensors(siglip_ctx * ctx, FILE * f, uint64_t n_tensors) {
    LOG_DEBUG("Lade %lu Tensoren...", (unsigned long)n_tensors);

    // Tensor-Infos lesen
    struct tensor_info {
        std::string name;
        uint32_t n_dims;
        std::vector<uint64_t> dims;
        uint32_t type;
        uint64_t offset;
    };
    std::vector<tensor_info> tensor_infos(n_tensors);

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

    // Berechne Gesamtgröße
    size_t total_size = 0;
    for (const auto & ti : tensor_infos) {
        size_t n_elements = 1;
        for (auto d : ti.dims) n_elements *= d;

        size_t element_size = 4; // Default F32
        if (ti.type == GGML_TYPE_F16) element_size = 2;
        else if (ti.type == GGML_TYPE_Q8_0) element_size = 1; // Approximation

        total_size += n_elements * element_size;
    }

    // GGML Kontext erstellen
    ggml_init_params ggml_params = {
        .mem_size   = total_size + 256 * 1024 * 1024, // Extra für Overhead
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ctx->ctx_data = ggml_init(ggml_params);
    if (!ctx->ctx_data) {
        set_error("Konnte GGML Kontext nicht erstellen");
        return false;
    }

    // Alignment auf 32 bytes für Daten-Start
    long current_pos = ftell(f);
    long alignment = 32;
    long padding = (alignment - (current_pos % alignment)) % alignment;
    fseek(f, padding, SEEK_CUR);
    long data_start = ftell(f);

    // Tensoren erstellen und laden
    ctx->tensors.blocks.resize(ctx->hparams.num_hidden_layers);

    for (const auto & ti : tensor_infos) {
        // GGML Tensor erstellen
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

        // Daten laden
        fseek(f, data_start + ti.offset, SEEK_SET);
        size_t bytes = ggml_nbytes(tensor);
        if (fread(tensor->data, 1, bytes, f) != bytes) {
            LOG_ERROR("Fehler beim Laden von Tensor %s", ti.name.c_str());
            return false;
        }

        // Tensor zuweisen
        const std::string & name = ti.name;

        if (name == "siglip.patch_embed.weight") {
            ctx->tensors.patch_embed_weight = tensor;
        } else if (name == "siglip.patch_embed.bias") {
            ctx->tensors.patch_embed_bias = tensor;
        } else if (name == "siglip.pos_embed") {
            ctx->tensors.pos_embed = tensor;
        } else if (name == "siglip.norm.weight") {
            ctx->tensors.norm_weight = tensor;
        } else if (name == "siglip.norm.bias") {
            ctx->tensors.norm_bias = tensor;
        } else if (name == "siglip.head.weight") {
            ctx->tensors.head_weight = tensor;
        } else if (name == "siglip.head.bias") {
            ctx->tensors.head_bias = tensor;
        } else if (name.find("siglip.blocks.") == 0) {
            // Block-Tensor parsen: siglip.blocks.{i}.{component}
            size_t dot1 = name.find('.', 14); // Nach "siglip.blocks."
            if (dot1 != std::string::npos) {
                int block_idx = std::stoi(name.substr(14, dot1 - 14));
                std::string component = name.substr(dot1 + 1);

                if (block_idx >= 0 && block_idx < ctx->hparams.num_hidden_layers) {
                    auto & block = ctx->tensors.blocks[block_idx];

                    if (component == "attn.q.weight") block.attn_q_weight = tensor;
                    else if (component == "attn.q.bias") block.attn_q_bias = tensor;
                    else if (component == "attn.k.weight") block.attn_k_weight = tensor;
                    else if (component == "attn.k.bias") block.attn_k_bias = tensor;
                    else if (component == "attn.v.weight") block.attn_v_weight = tensor;
                    else if (component == "attn.v.bias") block.attn_v_bias = tensor;
                    else if (component == "attn.out.weight") block.attn_out_weight = tensor;
                    else if (component == "attn.out.bias") block.attn_out_bias = tensor;
                    else if (component == "mlp.fc1.weight") block.mlp_fc1_weight = tensor;
                    else if (component == "mlp.fc1.bias") block.mlp_fc1_bias = tensor;
                    else if (component == "mlp.fc2.weight") block.mlp_fc2_weight = tensor;
                    else if (component == "mlp.fc2.bias") block.mlp_fc2_bias = tensor;
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
// API Implementation
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
        set_error("Konnte Datei nicht öffnen: %s", model_path);
        return nullptr;
    }

    // Magic prüfen
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GGUF_MAGIC) {
        set_error("Ungültige GGUF Magic: 0x%08X (erwartet 0x%08X)", magic, GGUF_MAGIC);
        fclose(f);
        return nullptr;
    }

    // Version prüfen
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

    // Default Hyperparameter (werden durch Metadaten überschrieben)
    ctx->hparams.model_type = SIGLIP_MODEL_VIT_B_16;
    ctx->hparams.hidden_size = 768;
    ctx->hparams.intermediate_size = 3072;
    ctx->hparams.num_attention_heads = 12;
    ctx->hparams.num_hidden_layers = 12;
    ctx->hparams.image_size = 224;
    ctx->hparams.patch_size = 16;
    ctx->hparams.num_patches = 196; // (224/16)^2
    ctx->hparams.layer_norm_eps = 1e-6f;

    // Default Preprocessing
    ctx->hparams.preprocess.target_size = 224;
    ctx->hparams.preprocess.mean[0] = 0.5f;
    ctx->hparams.preprocess.mean[1] = 0.5f;
    ctx->hparams.preprocess.mean[2] = 0.5f;
    ctx->hparams.preprocess.std[0] = 0.5f;
    ctx->hparams.preprocess.std[1] = 0.5f;
    ctx->hparams.preprocess.std[2] = 0.5f;
    ctx->hparams.preprocess.center_crop = false;
    ctx->hparams.preprocess.bicubic = true;

    // Metadaten lesen
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
            // Unbekannte Metadaten überspringen
            // Vereinfacht: Nur bekannte Typen
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

        if (callback) {
            callback(static_cast<float>(i + 1) / n_kv * 0.5f, user_data);
        }
    }

    // Modell-Typ bestimmen
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
// Modell-Info
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
// Bild-Handling
// ============================================================================

siglip_image * siglip_image_load(const char * path) {
    int width, height, channels;
    uint8_t * data = stbi_load(path, &width, &height, &channels, 3);

    if (!data) {
        set_error("Konnte Bild nicht laden: %s", path);
        return nullptr;
    }

    siglip_image * img = new siglip_image();
    img->data = data;
    img->width = width;
    img->height = height;
    img->channels = 3;

    return img;
}

siglip_image * siglip_image_from_raw(const uint8_t * data, int width, int height, int channels) {
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        set_error("Ungültige Bild-Parameter");
        return nullptr;
    }

    siglip_image * img = new siglip_image();
    size_t size = width * height * channels;
    img->data = new uint8_t[size];
    memcpy(img->data, data, size);
    img->width = width;
    img->height = height;
    img->channels = channels;

    return img;
}

void siglip_image_free(siglip_image * img) {
    if (img) {
        if (img->data) {
            stbi_image_free(img->data);
        }
        delete img;
    }
}

siglip_image * siglip_image_clone(const siglip_image * img) {
    if (!img) return nullptr;
    return siglip_image_from_raw(img->data, img->width, img->height, img->channels);
}

// ============================================================================
// Preprocessing
// ============================================================================

float * siglip_preprocess(const siglip_ctx * ctx, const siglip_image * img) {
    if (!ctx || !img) return nullptr;
    return siglip_preprocess_with_params(img, &ctx->hparams.preprocess);
}

float * siglip_preprocess_with_params(const siglip_image * img, const siglip_preprocess_params * params) {
    if (!img || !params) return nullptr;

    int target_size = params->target_size;

    // Resize
    std::vector<uint8_t> resized(target_size * target_size * 3);

    stbir_resize_uint8_linear(
        img->data, img->width, img->height, img->width * img->channels,
        resized.data(), target_size, target_size, target_size * 3,
        static_cast<stbir_pixel_layout>(STBIR_RGB)
    );

    // Normalize und zu CHW konvertieren
    float * output = new float[3 * target_size * target_size];

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                int src_idx = (y * target_size + x) * 3 + c;
                int dst_idx = c * target_size * target_size + y * target_size + x;

                float pixel = resized[src_idx] / 255.0f;
                output[dst_idx] = (pixel - params->mean[c]) / params->std[c];
            }
        }
    }

    return output;
}

void siglip_preprocess_free(float * preprocessed) {
    delete[] preprocessed;
}

// ============================================================================
// Encoding
// ============================================================================

// LayerNorm
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, weight);
    if (bias) {
        x = ggml_add(ctx, x, bias);
    }
    return x;
}

// GELU Activation
static ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_gelu(ctx, x);
}

// Self-Attention
static ggml_tensor * self_attention(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * q_w, ggml_tensor * q_b,
    ggml_tensor * k_w, ggml_tensor * k_b,
    ggml_tensor * v_w, ggml_tensor * v_b,
    ggml_tensor * out_w, ggml_tensor * out_b,
    int n_heads
) {
    int64_t n_tokens = x->ne[1];
    int64_t hidden = x->ne[0];
    int64_t head_dim = hidden / n_heads;

    // Q, K, V Projektionen
    ggml_tensor * q = ggml_mul_mat(ctx, q_w, x);
    if (q_b) q = ggml_add(ctx, q, q_b);

    ggml_tensor * k = ggml_mul_mat(ctx, k_w, x);
    if (k_b) k = ggml_add(ctx, k, k_b);

    ggml_tensor * v = ggml_mul_mat(ctx, v_w, x);
    if (v_b) v = ggml_add(ctx, v, v_b);

    // Reshape für Multi-Head Attention
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_tokens);

    // Permute: [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Attention Scores: Q @ K^T / sqrt(head_dim)
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf(static_cast<float>(head_dim)));

    // Softmax
    scores = ggml_soft_max(ctx, scores);

    // Attention Output: Scores @ V
    ggml_tensor * attn_out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_transpose(ctx, scores)));

    // Reshape zurück
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, hidden, n_tokens);

    // Output Projektion
    attn_out = ggml_mul_mat(ctx, out_w, attn_out);
    if (out_b) attn_out = ggml_add(ctx, attn_out, out_b);

    return attn_out;
}

// MLP Block
static ggml_tensor * mlp_block(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * fc1_w, ggml_tensor * fc1_b,
    ggml_tensor * fc2_w, ggml_tensor * fc2_b
) {
    x = ggml_mul_mat(ctx, fc1_w, x);
    if (fc1_b) x = ggml_add(ctx, x, fc1_b);
    x = gelu(ctx, x);
    x = ggml_mul_mat(ctx, fc2_w, x);
    if (fc2_b) x = ggml_add(ctx, x, fc2_b);
    return x;
}

// Build compute graph
static ggml_cgraph * build_graph(siglip_ctx * ctx, ggml_tensor * input) {
    const auto & hp = ctx->hparams;
    const auto & t = ctx->tensors;

    // Compute-Kontext erstellen
    size_t compute_size = 256 * 1024 * 1024; // 256 MB
    ggml_init_params params = {
        .mem_size   = compute_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    if (ctx->ctx_compute) {
        ggml_free(ctx->ctx_compute);
    }
    ctx->ctx_compute = ggml_init(params);

    // Input: [3, H, W] -> Patches: [hidden, num_patches]
    // Patch Embedding via Conv2D
    ggml_tensor * patches = ggml_conv_2d(
        ctx->ctx_compute,
        t.patch_embed_weight,
        input,
        hp.patch_size, hp.patch_size, // stride
        0, 0,                          // padding
        1, 1                           // dilation
    );

    // Reshape: [hidden, h_patches, w_patches] -> [hidden, num_patches]
    int h_patches = hp.image_size / hp.patch_size;
    int w_patches = hp.image_size / hp.patch_size;
    patches = ggml_reshape_2d(ctx->ctx_compute, patches, hp.hidden_size, hp.num_patches);

    // Add patch embedding bias
    if (t.patch_embed_bias) {
        patches = ggml_add(ctx->ctx_compute, patches, t.patch_embed_bias);
    }

    // Add positional embedding
    if (t.pos_embed) {
        patches = ggml_add(ctx->ctx_compute, patches, t.pos_embed);
    }

    // Transformer Blocks
    ggml_tensor * hidden = patches;

    for (int i = 0; i < hp.num_hidden_layers; i++) {
        const auto & block = t.blocks[i];

        // Pre-Norm Attention
        ggml_tensor * residual = hidden;
        hidden = layer_norm(ctx->ctx_compute, hidden, block.ln1_weight, block.ln1_bias, hp.layer_norm_eps);

        hidden = self_attention(
            ctx->ctx_compute, hidden,
            block.attn_q_weight, block.attn_q_bias,
            block.attn_k_weight, block.attn_k_bias,
            block.attn_v_weight, block.attn_v_bias,
            block.attn_out_weight, block.attn_out_bias,
            hp.num_attention_heads
        );

        hidden = ggml_add(ctx->ctx_compute, hidden, residual);

        // Pre-Norm MLP
        residual = hidden;
        hidden = layer_norm(ctx->ctx_compute, hidden, block.ln2_weight, block.ln2_bias, hp.layer_norm_eps);

        hidden = mlp_block(
            ctx->ctx_compute, hidden,
            block.mlp_fc1_weight, block.mlp_fc1_bias,
            block.mlp_fc2_weight, block.mlp_fc2_bias
        );

        hidden = ggml_add(ctx->ctx_compute, hidden, residual);
    }

    // Final LayerNorm
    hidden = layer_norm(ctx->ctx_compute, hidden, t.norm_weight, t.norm_bias, hp.layer_norm_eps);

    // MAP (Mean Attention Pooling) - vereinfacht als Mean Pooling
    // hidden: [hidden_size, num_patches] -> [hidden_size]
    ggml_tensor * pooled = ggml_pool_2d(
        ctx->ctx_compute,
        ggml_reshape_3d(ctx->ctx_compute, hidden, hp.hidden_size, hp.num_patches, 1),
        GGML_OP_POOL_AVG,
        hp.num_patches, 1,
        hp.num_patches, 1,
        0, 0
    );
    pooled = ggml_reshape_1d(ctx->ctx_compute, pooled, hp.hidden_size);

    // Optional: Projection Head
    if (t.head_weight) {
        pooled = ggml_mul_mat(ctx->ctx_compute, t.head_weight, pooled);
        if (t.head_bias) {
            pooled = ggml_add(ctx->ctx_compute, pooled, t.head_bias);
        }
    }

    // Graph erstellen
    ggml_cgraph * graph = ggml_new_graph(ctx->ctx_compute);
    ggml_build_forward_expand(graph, pooled);

    return graph;
}

siglip_embedding * siglip_encode(siglip_ctx * ctx, const siglip_image * img) {
    if (!ctx || !img) {
        set_error("Ungültige Parameter");
        return nullptr;
    }

    // Preprocessing
    float * preprocessed = siglip_preprocess(ctx, img);
    if (!preprocessed) {
        return nullptr;
    }

    siglip_embedding * result = siglip_encode_preprocessed(ctx, preprocessed);
    siglip_preprocess_free(preprocessed);

    return result;
}

siglip_embedding * siglip_encode_preprocessed(siglip_ctx * ctx, const float * preprocessed) {
    if (!ctx || !preprocessed) {
        set_error("Ungültige Parameter");
        return nullptr;
    }

    const auto & hp = ctx->hparams;

    // Input-Tensor erstellen
    ggml_tensor * input = ggml_new_tensor_3d(ctx->ctx_compute, GGML_TYPE_F32, hp.image_size, hp.image_size, 3);

    // Daten kopieren
    memcpy(input->data, preprocessed, 3 * hp.image_size * hp.image_size * sizeof(float));

    // Graph bauen und ausführen
    ggml_cgraph * graph = build_graph(ctx, input);

    // Backend initialisieren (falls noch nicht geschehen)
    if (!ctx->backend) {
        #ifdef GGML_USE_CUDA
        if (ctx->params.backend == SIGLIP_BACKEND_CUDA) {
            ctx->backend = ggml_backend_cuda_init(ctx->params.main_gpu);
        }
        #endif

        #ifdef GGML_USE_METAL
        if (ctx->params.backend == SIGLIP_BACKEND_METAL) {
            ctx->backend = ggml_backend_metal_init();
        }
        #endif

        if (!ctx->backend) {
            ctx->backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(ctx->backend, ctx->params.n_threads);
        }
    }

    // Graph Allocator
    if (!ctx->allocr) {
        ctx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    }

    ggml_gallocr_alloc_graph(ctx->allocr, graph);

    // Compute
    ggml_backend_graph_compute(ctx->backend, graph);

    // Ergebnis extrahieren
    ggml_tensor * output = graph->nodes[graph->n_nodes - 1];

    siglip_embedding * emb = new siglip_embedding();
    emb->size = hp.hidden_size;
    emb->batch_size = 1;
    emb->normalized = false;
    emb->data = new float[hp.hidden_size];

    ggml_backend_tensor_get(output, emb->data, 0, hp.hidden_size * sizeof(float));

    // Normalisieren falls gewünscht
    if (ctx->params.embed_format == SIGLIP_EMBED_NORMALIZED) {
        siglip_normalize(emb);
    }

    return emb;
}

siglip_embedding * siglip_encode_batch(siglip_ctx * ctx, const siglip_batch * batch) {
    if (!ctx || !batch || batch->n_images <= 0) {
        set_error("Ungültige Parameter");
        return nullptr;
    }

    const auto & hp = ctx->hparams;

    // Batch-Embeddings sammeln
    siglip_embedding * result = new siglip_embedding();
    result->size = hp.hidden_size;
    result->batch_size = batch->n_images;
    result->normalized = false;
    result->data = new float[hp.hidden_size * batch->n_images];

    for (int i = 0; i < batch->n_images; i++) {
        siglip_embedding * single = siglip_encode(ctx, batch->images[i]);
        if (single) {
            memcpy(result->data + i * hp.hidden_size, single->data, hp.hidden_size * sizeof(float));
            siglip_embedding_free(single);
        } else {
            // Fehler - mit Nullen füllen
            memset(result->data + i * hp.hidden_size, 0, hp.hidden_size * sizeof(float));
        }
    }

    return result;
}

void siglip_embedding_free(siglip_embedding * emb) {
    if (emb) {
        delete[] emb->data;
        delete emb;
    }
}

// ============================================================================
// Utilities
// ============================================================================

float siglip_cosine_similarity(const siglip_embedding * a, const siglip_embedding * b) {
    if (!a || !b || a->size != b->size) return 0.0f;
    return siglip_cosine_similarity_raw(a->data, b->data, a->size);
}

float siglip_cosine_similarity_raw(const float * a, const float * b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

void siglip_normalize(siglip_embedding * emb) {
    if (!emb || !emb->data) return;
    siglip_normalize_raw(emb->data, emb->size * emb->batch_size);
    emb->normalized = true;
}

void siglip_normalize_raw(float * data, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        for (int i = 0; i < size; i++) {
            data[i] /= norm;
        }
    }
}

int siglip_embedding_to_float(const siglip_embedding * emb, float * out, int size) {
    if (!emb || !out) return 0;
    int n = std::min(size, emb->size * emb->batch_size);
    memcpy(out, emb->data, n * sizeof(float));
    return n;
}

// ============================================================================
// Serialisierung
// ============================================================================

char * siglip_embedding_to_json(const siglip_embedding * emb) {
    if (!emb) return nullptr;

    std::string json = "{\"embedding\":[";
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

    char * result = new char[json.size() + 1];
    strcpy(result, json.c_str());
    return result;
}

uint8_t * siglip_embedding_to_binary(const siglip_embedding * emb, size_t * out_size) {
    if (!emb || !out_size) return nullptr;

    *out_size = emb->size * sizeof(float);
    uint8_t * result = new uint8_t[*out_size];
    memcpy(result, emb->data, *out_size);
    return result;
}

uint8_t * siglip_embedding_to_numpy(const siglip_embedding * emb, size_t * out_size) {
    if (!emb || !out_size) return nullptr;

    // NumPy .npy Format (vereinfacht)
    // Magic: \x93NUMPY
    // Version: 1.0
    // Header: {'descr': '<f4', 'fortran_order': False, 'shape': (size,)}

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(emb->size);
    header += ",), }";

    // Padding auf 64 bytes
    size_t header_len = header.size();
    size_t padding = 64 - ((10 + header_len) % 64);
    if (padding == 64) padding = 0;
    header.append(padding, ' ');
    header += '\n';

    size_t total_size = 10 + header.size() + emb->size * sizeof(float);
    uint8_t * result = new uint8_t[total_size];

    // Magic + Version
    result[0] = 0x93;
    result[1] = 'N';
    result[2] = 'U';
    result[3] = 'M';
    result[4] = 'P';
    result[5] = 'Y';
    result[6] = 1;  // Version major
    result[7] = 0;  // Version minor

    // Header length (little-endian uint16)
    uint16_t hlen = static_cast<uint16_t>(header.size());
    result[8] = hlen & 0xFF;
    result[9] = (hlen >> 8) & 0xFF;

    // Header
    memcpy(result + 10, header.c_str(), header.size());

    // Data
    memcpy(result + 10 + header.size(), emb->data, emb->size * sizeof(float));

    *out_size = total_size;
    return result;
}

// ============================================================================
// Fehlerbehandlung
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
// System-Info
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
