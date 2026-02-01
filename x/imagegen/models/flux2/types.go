//go:build mlx

// Package flux2 - Typdefinitionen und Konfiguration
// Enthält: TransformerConfig, TimestepEmbedder, TimeGuidanceEmbed, Modulation, FeedForward

package flux2

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// TransformerConfig holds Flux2 transformer configuration
type TransformerConfig struct {
	AttentionHeadDim         int32   `json:"attention_head_dim"`         // 128
	AxesDimsRoPE             []int32 `json:"axes_dims_rope"`             // [32, 32, 32, 32]
	Eps                      float32 `json:"eps"`                        // 1e-6
	GuidanceEmbeds           bool    `json:"guidance_embeds"`            // false for Klein
	InChannels               int32   `json:"in_channels"`                // 128
	JointAttentionDim        int32   `json:"joint_attention_dim"`        // 7680
	MLPRatio                 float32 `json:"mlp_ratio"`                  // 3.0
	NumAttentionHeads        int32   `json:"num_attention_heads"`        // 24
	NumLayers                int32   `json:"num_layers"`                 // 5
	NumSingleLayers          int32   `json:"num_single_layers"`          // 20
	PatchSize                int32   `json:"patch_size"`                 // 1
	RopeTheta                int32   `json:"rope_theta"`                 // 2000
	TimestepGuidanceChannels int32   `json:"timestep_guidance_channels"` // 256
}

// Computed dimensions
func (c *TransformerConfig) InnerDim() int32 {
	return c.NumAttentionHeads * c.AttentionHeadDim // 24 * 128 = 3072
}

func (c *TransformerConfig) MLPHiddenDim() int32 {
	return int32(float32(c.InnerDim()) * c.MLPRatio) // 3072 * 3.0 = 9216
}

// TimestepEmbedder creates timestep embeddings
// Weight names: time_guidance_embed.timestep_embedder.linear_1.weight, linear_2.weight
type TimestepEmbedder struct {
	Linear1  nn.LinearLayer `weight:"linear_1"`
	Linear2  nn.LinearLayer `weight:"linear_2"`
	EmbedDim int32          // 256
}

// Forward creates sinusoidal embeddings and projects them
func (t *TimestepEmbedder) Forward(timesteps *mlx.Array) *mlx.Array {
	half := t.EmbedDim / 2
	freqs := make([]float32, half)
	for i := int32(0); i < half; i++ {
		freqs[i] = float32(math.Exp(-math.Log(10000.0) * float64(i) / float64(half)))
	}
	freqsArr := mlx.NewArray(freqs, []int32{1, half})

	// timesteps: [B] -> [B, 1]
	tExpanded := mlx.ExpandDims(timesteps, 1)
	// args: [B, half]
	args := mlx.Mul(tExpanded, freqsArr)

	// [cos(args), sin(args)] -> [B, embed_dim]
	sinEmbed := mlx.Concatenate([]*mlx.Array{mlx.Cos(args), mlx.Sin(args)}, 1)

	// MLP: linear_1 -> silu -> linear_2
	h := t.Linear1.Forward(sinEmbed)
	h = mlx.SiLU(h)
	return t.Linear2.Forward(h)
}

// TimeGuidanceEmbed wraps the timestep embedder
// Weight names: time_guidance_embed.timestep_embedder.*
type TimeGuidanceEmbed struct {
	TimestepEmbedder *TimestepEmbedder `weight:"timestep_embedder"`
}

// Forward computes timestep embeddings
func (t *TimeGuidanceEmbed) Forward(timesteps *mlx.Array) *mlx.Array {
	return t.TimestepEmbedder.Forward(timesteps)
}

// Modulation computes adaptive modulation parameters
// Weight names: double_stream_modulation_img.linear.weight, etc.
type Modulation struct {
	Linear nn.LinearLayer `weight:"linear"`
}

// Forward computes modulation parameters
func (m *Modulation) Forward(temb *mlx.Array) *mlx.Array {
	h := mlx.SiLU(temb)
	return m.Linear.Forward(h)
}

// FeedForward implements SwiGLU MLP
// Weight names: transformer_blocks.N.ff.linear_in.weight, linear_out.weight
type FeedForward struct {
	LinearIn  nn.LinearLayer `weight:"linear_in"`
	LinearOut nn.LinearLayer `weight:"linear_out"`
}

// Forward applies SwiGLU MLP
func (ff *FeedForward) Forward(x *mlx.Array) *mlx.Array {
	// LinearIn outputs 2x hidden dim for SwiGLU
	h := ff.LinearIn.Forward(x)
	shape := h.Shape()
	half := shape[len(shape)-1] / 2

	// Split into gate and up
	gate := mlx.Slice(h, []int32{0, 0, 0}, []int32{shape[0], shape[1], half})
	up := mlx.Slice(h, []int32{0, 0, half}, []int32{shape[0], shape[1], shape[2]})

	// SwiGLU: silu(gate) * up
	h = mlx.Mul(mlx.SiLU(gate), up)
	return ff.LinearOut.Forward(h)
}

// NormOut implements the output normalization with modulation
// Weight names: norm_out.linear.weight
type NormOut struct {
	Linear nn.LinearLayer `weight:"linear"`
}

// Forward computes final modulated output
func (n *NormOut) Forward(x *mlx.Array, temb *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	dim := shape[2]

	// Modulation: temb -> silu -> linear -> [shift, scale]
	mod := mlx.SiLU(temb)
	mod = n.Linear.Forward(mod)

	// Split into scale and shift (diffusers order: scale first, shift second)
	scale := mlx.Slice(mod, []int32{0, 0}, []int32{B, dim})
	shift := mlx.Slice(mod, []int32{0, dim}, []int32{B, 2 * dim})
	shift = mlx.ExpandDims(shift, 1)
	scale = mlx.ExpandDims(scale, 1)

	// Modulate with RMSNorm
	return modulateLayerNorm(x, shift, scale)
}
