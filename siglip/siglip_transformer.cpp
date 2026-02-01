/**
 * siglip_transformer.cpp - SigLIP Transformer-Komponenten
 *
 * Dieser Teil implementiert die Grundbausteine des Vision Transformers:
 * - layer_norm() - Layer Normalization
 * - gelu() - GELU Aktivierungsfunktion
 * - self_attention() - Multi-Head Self-Attention
 * - mlp_block() - Feed-Forward Network
 */

#include "siglip.h"

#include <cmath>

// GGML Headers
#include "ggml.h"

// ============================================================================
// Transformer-Komponenten - Grundbausteine
// ============================================================================

/**
 * Layer Normalization
 * Normalisiert den Input und skaliert/verschiebt mit weight/bias
 *
 * @param ctx GGML Kontext
 * @param x Input Tensor
 * @param weight Skalierungsgewichte (gamma)
 * @param bias Verschiebung (beta), kann nullptr sein
 * @param eps Epsilon fuer numerische Stabilitaet
 * @return Normalisierter Tensor
 */
ggml_tensor * siglip_layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float eps
) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, weight);
    if (bias) {
        x = ggml_add(ctx, x, bias);
    }
    return x;
}

/**
 * GELU Activation Function
 * Gaussian Error Linear Unit - sanftere Alternative zu ReLU
 *
 * @param ctx GGML Kontext
 * @param x Input Tensor
 * @return Aktivierter Tensor
 */
ggml_tensor * siglip_gelu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_gelu(ctx, x);
}

/**
 * Multi-Head Self-Attention
 * Kernkomponente des Transformers - ermoeglicht Aufmerksamkeit ueber alle Positionen
 *
 * @param ctx GGML Kontext
 * @param x Input Tensor [hidden, n_tokens]
 * @param q_w, q_b Query Projektion
 * @param k_w, k_b Key Projektion
 * @param v_w, v_b Value Projektion
 * @param out_w, out_b Output Projektion
 * @param n_heads Anzahl Attention Heads
 * @return Output Tensor [hidden, n_tokens]
 */
ggml_tensor * siglip_self_attention(
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

    // ====================================
    // Q, K, V Projektionen
    // ====================================
    ggml_tensor * q = ggml_mul_mat(ctx, q_w, x);
    if (q_b) q = ggml_add(ctx, q, q_b);

    ggml_tensor * k = ggml_mul_mat(ctx, k_w, x);
    if (k_b) k = ggml_add(ctx, k, k_b);

    ggml_tensor * v = ggml_mul_mat(ctx, v_w, x);
    if (v_b) v = ggml_add(ctx, v, v_b);

    // ====================================
    // Reshape fuer Multi-Head Attention
    // [hidden, n_tokens] -> [head_dim, n_heads, n_tokens]
    // ====================================
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_tokens);

    // Permute: [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // ====================================
    // Attention Scores berechnen
    // Scores = Q @ K^T / sqrt(head_dim)
    // ====================================
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf(static_cast<float>(head_dim)));

    // Softmax ueber die letzte Dimension
    scores = ggml_soft_max(ctx, scores);

    // ====================================
    // Attention Output
    // Output = Scores @ V
    // ====================================
    ggml_tensor * attn_out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_transpose(ctx, scores)));

    // Reshape zurueck: [head_dim, n_tokens, n_heads] -> [hidden, n_tokens]
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, hidden, n_tokens);

    // ====================================
    // Output Projektion
    // ====================================
    attn_out = ggml_mul_mat(ctx, out_w, attn_out);
    if (out_b) attn_out = ggml_add(ctx, attn_out, out_b);

    return attn_out;
}

/**
 * MLP Block (Feed-Forward Network)
 * Zwei lineare Schichten mit GELU Aktivierung dazwischen
 *
 * @param ctx GGML Kontext
 * @param x Input Tensor
 * @param fc1_w, fc1_b Erste lineare Schicht (Expansion)
 * @param fc2_w, fc2_b Zweite lineare Schicht (Projektion)
 * @return Output Tensor
 */
ggml_tensor * siglip_mlp_block(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * fc1_w, ggml_tensor * fc1_b,
    ggml_tensor * fc2_w, ggml_tensor * fc2_b
) {
    // FC1: hidden -> intermediate (Expansion)
    x = ggml_mul_mat(ctx, fc1_w, x);
    if (fc1_b) x = ggml_add(ctx, x, fc1_b);

    // GELU Aktivierung
    x = siglip_gelu(ctx, x);

    // FC2: intermediate -> hidden (Projektion)
    x = ggml_mul_mat(ctx, fc2_w, x);
    if (fc2_b) x = ggml_add(ctx, x, fc2_b);

    return x;
}
