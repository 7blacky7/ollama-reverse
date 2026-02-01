//go:build mlx

// Package zimage - Typdefinitionen und Konfiguration
// Enthält: TransformerConfig, TimestepEmbedder, XEmbedder, CapEmbedder, FeedForward

package zimage

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// TransformerConfig holds Z-Image transformer configuration
type TransformerConfig struct {
	Dim            int32   `json:"dim"`
	NHeads         int32   `json:"n_heads"`
	NKVHeads       int32   `json:"n_kv_heads"`
	NLayers        int32   `json:"n_layers"`
	NRefinerLayers int32   `json:"n_refiner_layers"`
	InChannels     int32   `json:"in_channels"`
	PatchSize      int32   `json:"-"` // Computed from AllPatchSize
	CapFeatDim     int32   `json:"cap_feat_dim"`
	NormEps        float32 `json:"norm_eps"`
	RopeTheta      float32 `json:"rope_theta"`
	TScale         float32 `json:"t_scale"`
	QKNorm         bool    `json:"qk_norm"`
	AxesDims       []int32 `json:"axes_dims"`
	AxesLens       []int32 `json:"axes_lens"`
	AllPatchSize   []int32 `json:"all_patch_size"` // JSON array, PatchSize = first element
}

// TimestepEmbedder creates sinusoidal timestep embeddings
// Output dimension is 256 (fixed), used for AdaLN modulation
type TimestepEmbedder struct {
	Linear1       nn.LinearLayer `weight:"mlp.0"`
	Linear2       nn.LinearLayer `weight:"mlp.2"`
	FreqEmbedSize int32          // 256 (computed)
}

// Forward computes timestep embeddings -> [B, 256]
func (te *TimestepEmbedder) Forward(t *mlx.Array) *mlx.Array {
	// t: [B] timesteps

	// Create sinusoidal embedding
	half := te.FreqEmbedSize / 2

	// freqs = exp(-log(10000) * arange(half) / half)
	freqs := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(half)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{1, half})

	// t[:, None] * freqs[None, :] -> [B, half]
	tExpanded := mlx.ExpandDims(t, 1) // [B, 1]
	args := mlx.Mul(tExpanded, freqsArr)

	// embedding = [cos(args), sin(args)] -> [B, 256]
	cosArgs := mlx.Cos(args)
	sinArgs := mlx.Sin(args)
	embedding := mlx.Concatenate([]*mlx.Array{cosArgs, sinArgs}, 1)

	// MLP: linear1 -> silu -> linear2
	h := te.Linear1.Forward(embedding)
	h = mlx.SiLU(h)
	h = te.Linear2.Forward(h)

	return h
}

// XEmbedder embeds image patches to model dimension
type XEmbedder struct {
	Linear nn.LinearLayer `weight:"2-1"`
}

// Forward embeds patchified image latents
func (xe *XEmbedder) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, L, in_channels * 4] -> [B, L, dim]
	return xe.Linear.Forward(x)
}

// CapEmbedder projects caption features to model dimension
type CapEmbedder struct {
	Norm     *nn.RMSNorm    `weight:"0"`
	Linear   nn.LinearLayer `weight:"1"`
	PadToken *mlx.Array     // loaded separately at root level
}

// Forward projects caption embeddings: [B, L, cap_feat_dim] -> [B, L, dim]
func (ce *CapEmbedder) Forward(capFeats *mlx.Array) *mlx.Array {
	// RMSNorm on last axis (uses 1e-6)
	h := ce.Norm.Forward(capFeats, 1e-6)
	// Linear projection
	return ce.Linear.Forward(h)
}

// FeedForward implements SwiGLU FFN
type FeedForward struct {
	W1     nn.LinearLayer `weight:"w1"` // gate projection
	W2     nn.LinearLayer `weight:"w2"` // down projection
	W3     nn.LinearLayer `weight:"w3"` // up projection
	OutDim int32          // computed from W2
}

// Forward applies SwiGLU: silu(W1(x)) * W3(x), then W2
func (ff *FeedForward) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	// Reshape for matmul
	x = mlx.Reshape(x, B*L, D)

	gate := ff.W1.Forward(x)
	gate = mlx.SiLU(gate)
	up := ff.W3.Forward(x)
	h := mlx.Mul(gate, up)
	out := ff.W2.Forward(h)

	return mlx.Reshape(out, B, L, ff.OutDim)
}
