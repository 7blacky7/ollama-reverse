/**
 * MODUL: clip_wrapper
 * ZWECK: C-Wrapper fuer CLIP Vision Encoder (llama.cpp Integration)
 * INPUT: Modell-Pfad, Rohbilddaten (JPEG/PNG/etc.)
 * OUTPUT: Float-Embeddings, Modell-Metadaten
 * NEBENEFFEKTE: Modell-Laden, GPU-Speicherallokation
 * ABHAENGIGKEITEN: clip.cpp (extern), clip-impl.h, stb_image (extern)
 * HINWEISE: Thread-sicher pro Context, CGO-kompatibel durch C-Linkage
 */

#include "clip_wrapper.h"

// Upstream CLIP-Includes
#include "../../llama.cpp.upstream/tools/mtmd/clip.h"
#include "../../llama.cpp.upstream/tools/mtmd/clip-impl.h"

// Bilddekodierung (stb_image ist in llama.cpp enthalten)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "../../llama.cpp.upstream/common/stb_image.h"

#include <cstring>
#include <cstdlib>

/* ============================================================================
 * Interne Hilfsstrukturen
 * ============================================================================ */

// Wrapper-Context mit zusaetzlichen Metadaten
struct clip_wrapper_ctx {
    struct clip_ctx* ctx;           // Originaler CLIP-Context
    char model_name[256];           // Modell-Name Cache
    int32_t embedding_dim;          // Embedding-Dimension Cache
    int32_t image_size;             // Bildgroesse Cache
};

/* ============================================================================
 * Interne Hilfsfunktionen
 * ============================================================================ */

// Rohbilddaten zu clip_image_u8 dekodieren
static int decode_image_data(
    const uint8_t* data,
    size_t data_size,
    struct clip_image_u8* img_out
) {
    int width, height, channels;

    // stb_image dekodiert automatisch JPEG, PNG, BMP, etc.
    unsigned char* pixels = stbi_load_from_memory(
        data,
        static_cast<int>(data_size),
        &width,
        &height,
        &channels,
        3  // Immer RGB anfordern
    );

    if (!pixels) {
        return CLIP_ERR_DECODE;
    }

    // Pixel in clip_image_u8 kopieren
    clip_build_img_from_pixels(pixels, width, height, img_out);

    stbi_image_free(pixels);
    return CLIP_OK;
}

// Embedding-Dimension aus Context ermitteln
static int32_t get_embedding_dimension(struct clip_ctx* ctx) {
    return clip_n_mmproj_embd(ctx);
}

/* ============================================================================
 * Oeffentliche API - Initialisierung
 * ============================================================================ */

extern "C" {

clip_init_params clip_wrapper_default_params(void) {
    clip_init_params params;
    params.n_threads = 4;
    params.n_gpu_layers = -1;  // Alle Layer auf GPU
    params.main_gpu = 0;
    params.use_mmap = 1;
    params.use_mlock = 0;
    return params;
}

clip_ctx* clip_wrapper_init(const char* model_path, clip_init_params params) {
    if (!model_path) {
        return nullptr;
    }

    // Upstream-Parameter vorbereiten
    struct clip_context_params ctx_params;
    ctx_params.use_gpu = (params.n_gpu_layers != 0);
    ctx_params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    ctx_params.image_min_tokens = 0;
    ctx_params.image_max_tokens = 0;
    ctx_params.warmup = true;
    ctx_params.cb_eval = nullptr;
    ctx_params.cb_eval_user_data = nullptr;

    // Modell laden (Vision-Context)
    struct clip_init_result result = clip_init(model_path, ctx_params);

    // Vision-Context zurueckgeben (Audio wird nicht benoetigt)
    if (result.ctx_a) {
        clip_free(result.ctx_a);
    }

    return result.ctx_v;
}

void clip_wrapper_free(clip_ctx* ctx) {
    if (ctx) {
        clip_free(ctx);
    }
}

/* ============================================================================
 * Oeffentliche API - Encoding
 * ============================================================================ */

int clip_wrapper_encode_image(
    clip_ctx* ctx,
    const uint8_t* image_data,
    size_t image_size,
    float* embedding,
    int32_t embedding_dim
) {
    // Eingabe-Validierung
    if (!ctx) {
        return CLIP_ERR_NULL_CTX;
    }
    if (!image_data || image_size == 0) {
        return CLIP_ERR_NULL_IMAGE;
    }
    if (!embedding) {
        return CLIP_ERR_ALLOC;
    }

    // Embedding-Dimension pruefen
    int32_t expected_dim = get_embedding_dimension(ctx);
    if (embedding_dim < expected_dim) {
        return CLIP_ERR_ALLOC;
    }

    // Bild dekodieren
    struct clip_image_u8* img_u8 = clip_image_u8_init();
    if (!img_u8) {
        return CLIP_ERR_ALLOC;
    }

    int decode_result = decode_image_data(image_data, image_size, img_u8);
    if (decode_result != CLIP_OK) {
        clip_image_u8_free(img_u8);
        return decode_result;
    }

    // Bild vorverarbeiten (Normalisierung, Resize, etc.)
    struct clip_image_f32_batch* img_batch = clip_image_f32_batch_init();
    if (!img_batch) {
        clip_image_u8_free(img_u8);
        return CLIP_ERR_ALLOC;
    }

    bool preprocess_ok = clip_image_preprocess(ctx, img_u8, img_batch);
    clip_image_u8_free(img_u8);

    if (!preprocess_ok) {
        clip_image_f32_batch_free(img_batch);
        return CLIP_ERR_ENCODE;
    }

    // Encoding durchfuehren (erstes Bild im Batch)
    size_t n_images = clip_image_f32_batch_n_images(img_batch);
    if (n_images == 0) {
        clip_image_f32_batch_free(img_batch);
        return CLIP_ERR_ENCODE;
    }

    struct clip_image_f32* img_f32 = clip_image_f32_get_img(img_batch, 0);
    bool encode_ok = clip_image_encode(ctx, 4, img_f32, embedding);

    clip_image_f32_batch_free(img_batch);

    return encode_ok ? CLIP_OK : CLIP_ERR_ENCODE;
}

int clip_wrapper_encode_batch(
    clip_ctx* ctx,
    const uint8_t** images,
    const size_t* image_sizes,
    int32_t batch_size,
    float* embeddings,
    int32_t embedding_dim
) {
    if (!ctx) {
        return CLIP_ERR_NULL_CTX;
    }
    if (!images || !image_sizes || batch_size <= 0) {
        return CLIP_ERR_NULL_IMAGE;
    }

    // Jedes Bild einzeln verarbeiten (Upstream unterstuetzt batch_size=1)
    int32_t emb_dim = get_embedding_dimension(ctx);

    for (int32_t i = 0; i < batch_size; i++) {
        int result = clip_wrapper_encode_image(
            ctx,
            images[i],
            image_sizes[i],
            embeddings + (i * emb_dim),
            embedding_dim
        );

        if (result != CLIP_OK) {
            return result;
        }
    }

    return CLIP_OK;
}

/* ============================================================================
 * Oeffentliche API - Metadaten
 * ============================================================================ */

clip_model_info clip_wrapper_get_model_info(clip_ctx* ctx) {
    clip_model_info info;
    info.name = nullptr;
    info.embedding_dim = 0;
    info.image_size = 0;

    if (ctx) {
        info.embedding_dim = get_embedding_dimension(ctx);
        info.image_size = clip_get_image_size(ctx);
    }

    return info;
}

int32_t clip_wrapper_get_embedding_dim(clip_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return get_embedding_dimension(ctx);
}

int32_t clip_wrapper_get_image_size(clip_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return clip_get_image_size(ctx);
}

} // extern "C"
