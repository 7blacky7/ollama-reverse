/**
 * MODUL: nomic_inference.cpp
 * ZWECK: Vision Transformer Forward Pass fuer Nomic Embed Vision
 * INPUT: nomic_ctx, vorverarbeitete Bilddaten [3, H, W]
 * OUTPUT: Embedding-Vektor [768], L2-normalisiert
 * NEBENEFFEKTE: GPU/CPU Compute, temporaere Allokationen
 * ABHAENGIGKEITEN: nomic_internal.h, ggml
 * HINWEISE: ViT mit CLS-Token Pooling, SwiGLU Aktivierung
 *
 * Architektur:
 * 1. Patch Embedding: [3,384,384] -> [729,768] (Conv2D als Linear)
 * 2. CLS Token + Position Embedding: [730,768]
 * 3. 12x Transformer Block (Attention + SwiGLU MLP)
 * 4. Final LayerNorm
 * 5. CLS Token extrahieren -> [768]
 */

#include "nomic_internal.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// GGML Graph Building - Transformer Komponenten
// ============================================================================

/** Layer Normalization */
static ggml_tensor * build_layer_norm(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * weight, ggml_tensor * bias, float eps
) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, weight);
    if (bias) x = ggml_add(ctx, x, bias);
    return x;
}

/** Multi-Head Self-Attention */
static ggml_tensor * build_attention(
    ggml_context * ctx, ggml_tensor * x,
    const nomic_layer & layer, int n_heads, float eps
) {
    int64_t n_tokens = x->ne[1];
    int64_t hidden = x->ne[0];
    int64_t head_dim = hidden / n_heads;

    // Q, K, V Projektionen
    ggml_tensor * q = ggml_mul_mat(ctx, layer.q_weight, x);
    if (layer.q_bias) q = ggml_add(ctx, q, layer.q_bias);

    ggml_tensor * k = ggml_mul_mat(ctx, layer.k_weight, x);
    if (layer.k_bias) k = ggml_add(ctx, k, layer.k_bias);

    ggml_tensor * v = ggml_mul_mat(ctx, layer.v_weight, x);
    if (layer.v_bias) v = ggml_add(ctx, v, layer.v_bias);

    // Reshape fuer Multi-Head: [hidden,tokens] -> [head_dim,heads,tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_tokens);

    // Permute: [head_dim,heads,tokens] -> [head_dim,tokens,heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Attention Scores: Q @ K^T / sqrt(d)
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));
    scores = ggml_soft_max(ctx, scores);

    // Attention Output: Scores @ V
    ggml_tensor * out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_transpose(ctx, scores)));

    // Reshape zurueck: [head_dim,tokens,heads] -> [hidden,tokens]
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont(ctx, out);
    out = ggml_reshape_2d(ctx, out, hidden, n_tokens);

    // Output Projektion
    out = ggml_mul_mat(ctx, layer.o_weight, out);
    if (layer.o_bias) out = ggml_add(ctx, out, layer.o_bias);

    return out;
}

/** SwiGLU MLP Block */
static ggml_tensor * build_swiglu_mlp(
    ggml_context * ctx, ggml_tensor * x, const nomic_layer & layer
) {
    // SwiGLU: out = down(silu(gate(x)) * up(x))
    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate_weight, x);
    if (layer.ffn_gate_bias) gate = ggml_add(ctx, gate, layer.ffn_gate_bias);
    gate = ggml_silu(ctx, gate);

    ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_weight, x);
    if (layer.ffn_up_bias) up = ggml_add(ctx, up, layer.ffn_up_bias);

    ggml_tensor * hidden = ggml_mul(ctx, gate, up);

    ggml_tensor * down = ggml_mul_mat(ctx, layer.ffn_down_weight, hidden);
    if (layer.ffn_down_bias) down = ggml_add(ctx, down, layer.ffn_down_bias);

    return down;
}

/** Standard GELU MLP Block (Fallback) */
static ggml_tensor * build_gelu_mlp(
    ggml_context * ctx, ggml_tensor * x, const nomic_layer & layer
) {
    // Standard: out = down(gelu(up(x)))
    ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_weight, x);
    if (layer.ffn_up_bias) up = ggml_add(ctx, up, layer.ffn_up_bias);
    up = ggml_gelu(ctx, up);

    ggml_tensor * down = ggml_mul_mat(ctx, layer.ffn_down_weight, up);
    if (layer.ffn_down_bias) down = ggml_add(ctx, down, layer.ffn_down_bias);

    return down;
}

// ============================================================================
// Forward Pass
// ============================================================================

bool nomic_forward(nomic_ctx * ctx, const float * input, float * output) {
    const nomic_hparams & hp = ctx->hparams;
    const int n_patches = hp.num_patches;
    const int hidden = hp.hidden_size;
    const int n_layers = hp.num_hidden_layers;
    const int n_heads = hp.num_attention_heads;
    const float eps = hp.layer_norm_eps;

    NOMIC_LOG_DBG("Forward: %d patches, %d hidden, %d layers", n_patches, hidden, n_layers);

    // Compute-Kontext erstellen (grosszuegig: 512MB fuer Zwischenwerte)
    size_t compute_size = 512 * 1024 * 1024;
    ggml_init_params params = {compute_size, nullptr, false};
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        nomic_set_error("Compute-Kontext konnte nicht erstellt werden");
        return false;
    }

    // Input Tensor: [3, H, W] -> flatten fuer Patch Embedding
    int img_size = hp.image_size;
    int patch_size = hp.patch_size;
    int n_channels = 3;
    int patch_dim = n_channels * patch_size * patch_size;  // 3*14*14 = 588

    // Input-Tensor erstellen
    ggml_tensor * img = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, img_size, img_size, n_channels);
    memcpy(img->data, input, n_channels * img_size * img_size * sizeof(float));

    // ========================================
    // 1. Patch Embedding
    // ========================================
    // Patches extrahieren und linear projizieren
    // Vereinfacht: Assume patch_embed_weight ist [hidden, patch_dim]
    ggml_tensor * patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, patch_dim, n_patches);

    // Patches aus Bild extrahieren (manuell - GGML hat keine im2col fuer ViT)
    float * patch_data = (float*)patches->data;
    const float * img_data = input;

    int grid = img_size / patch_size;  // 27
    for (int py = 0; py < grid; py++) {
        for (int px = 0; px < grid; px++) {
            int patch_idx = py * grid + px;
            float * dst = patch_data + patch_idx * patch_dim;

            for (int c = 0; c < n_channels; c++) {
                for (int y = 0; y < patch_size; y++) {
                    for (int x = 0; x < patch_size; x++) {
                        int img_y = py * patch_size + y;
                        int img_x = px * patch_size + x;
                        int src_idx = c * img_size * img_size + img_y * img_size + img_x;
                        int dst_idx = c * patch_size * patch_size + y * patch_size + x;
                        dst[dst_idx] = img_data[src_idx];
                    }
                }
            }
        }
    }

    // Linear Projektion: [patch_dim, n_patches] -> [hidden, n_patches]
    ggml_tensor * x = ggml_mul_mat(gctx, ctx->tensors.patch_embed_weight, patches);
    if (ctx->tensors.patch_embed_bias) {
        x = ggml_add(gctx, x, ctx->tensors.patch_embed_bias);
    }

    // ========================================
    // 2. CLS Token + Position Embedding
    // ========================================
    // CLS Token prependen: [hidden,1] + [hidden,n_patches] -> [hidden,n_patches+1]
    if (ctx->tensors.cls_token) {
        ggml_tensor * cls = ggml_reshape_2d(gctx, ctx->tensors.cls_token, hidden, 1);
        x = ggml_concat(gctx, cls, x, 1);  // dim=1 fuer tokens
    }

    // Position Embedding addieren
    if (ctx->tensors.pos_embed) {
        x = ggml_add(gctx, x, ctx->tensors.pos_embed);
    }

    // ========================================
    // 3. Transformer Blocks
    // ========================================
    for (int l = 0; l < n_layers; l++) {
        const nomic_layer & layer = ctx->tensors.layers[l];

        // Pre-Norm Attention
        ggml_tensor * attn_in = build_layer_norm(gctx, x, layer.ln1_weight, layer.ln1_bias, eps);
        ggml_tensor * attn_out = build_attention(gctx, attn_in, layer, n_heads, eps);
        x = ggml_add(gctx, x, attn_out);  // Residual

        // Pre-Norm MLP
        ggml_tensor * mlp_in = build_layer_norm(gctx, x, layer.ln2_weight, layer.ln2_bias, eps);
        ggml_tensor * mlp_out;

        // SwiGLU wenn gate vorhanden, sonst GELU
        if (layer.ffn_gate_weight) {
            mlp_out = build_swiglu_mlp(gctx, mlp_in, layer);
        } else {
            mlp_out = build_gelu_mlp(gctx, mlp_in, layer);
        }
        x = ggml_add(gctx, x, mlp_out);  // Residual
    }

    // ========================================
    // 4. Final LayerNorm
    // ========================================
    if (ctx->tensors.post_ln_weight) {
        x = build_layer_norm(gctx, x, ctx->tensors.post_ln_weight, ctx->tensors.post_ln_bias, eps);
    }

    // ========================================
    // 5. CLS Token extrahieren (Index 0)
    // ========================================
    ggml_tensor * cls_out = ggml_view_1d(gctx, x, hidden, 0);

    // ========================================
    // Compute-Graph ausfuehren
    // ========================================
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, cls_out);

    // Allocator fuer Graph
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // Output kopieren
    memcpy(output, cls_out->data, hidden * sizeof(float));

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(gctx);

    return true;
}

// ============================================================================
// Oeffentliche API - Encoding
// ============================================================================

nomic_embedding * nomic_encode_preprocessed(nomic_ctx * ctx, const float * preprocessed) {
    if (!ctx || !preprocessed) {
        nomic_set_error("Ungueltige Parameter");
        return nullptr;
    }

    const nomic_hparams & hp = ctx->hparams;

    // Embedding allokieren
    nomic_embedding * emb = new nomic_embedding();
    emb->dim = hp.hidden_size;
    emb->batch_size = 1;
    emb->normalized = false;
    emb->data = new float[hp.hidden_size];

    // Forward Pass
    if (!nomic_forward(ctx, preprocessed, emb->data)) {
        delete[] emb->data;
        delete emb;
        return nullptr;
    }

    // L2 Normalisierung
    nomic_normalize(emb->data, hp.hidden_size);
    emb->normalized = true;

    return emb;
}

nomic_embedding * nomic_encode_image(
    nomic_ctx * ctx,
    const uint8_t * image_data,
    int width, int height
) {
    if (!ctx || !image_data) {
        nomic_set_error("Ungueltige Parameter");
        return nullptr;
    }

    const nomic_hparams & hp = ctx->hparams;
    int target_size = hp.image_size;
    int channels = 3;

    // Preprocessing: Resize + Normalize
    size_t out_size = channels * target_size * target_size;
    float * preprocessed = new float[out_size];

    // ImageNet Normalisierung
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std_[3] = {0.229f, 0.224f, 0.225f};

    float scale_x = (float)width / target_size;
    float scale_y = (float)height / target_size;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                // Bilineare Interpolation
                float src_x = x * scale_x;
                float src_y = y * scale_y;
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = std::min(x0 + 1, width - 1);
                int y1 = std::min(y0 + 1, height - 1);
                float fx = src_x - x0;
                float fy = src_y - y0;

                // Pixel holen (HWC Input)
                float p00 = image_data[(y0 * width + x0) * channels + c] / 255.0f;
                float p10 = image_data[(y0 * width + x1) * channels + c] / 255.0f;
                float p01 = image_data[(y1 * width + x0) * channels + c] / 255.0f;
                float p11 = image_data[(y1 * width + x1) * channels + c] / 255.0f;

                float val = p00*(1-fx)*(1-fy) + p10*fx*(1-fy) + p01*(1-fx)*fy + p11*fx*fy;

                // Normalisieren (CHW Output)
                val = (val - mean[c]) / std_[c];
                preprocessed[c * target_size * target_size + y * target_size + x] = val;
            }
        }
    }

    // Encoding
    nomic_embedding * emb = nomic_encode_preprocessed(ctx, preprocessed);
    delete[] preprocessed;

    return emb;
}

void nomic_embedding_free(nomic_embedding * emb) {
    if (emb) {
        delete[] emb->data;
        delete emb;
    }
}

// ============================================================================
// Utilities
// ============================================================================

void nomic_normalize(float * data, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrtf(norm + 1e-12f);

    for (int i = 0; i < size; i++) {
        data[i] /= norm;
    }
}

float nomic_cosine_similarity(const float * a, const float * b, int size) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}
