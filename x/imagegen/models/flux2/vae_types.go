//go:build mlx

// vae_types.go - Konfiguration und Hauptstruktur des Flux2 VAE
// Enthält VAEConfig und die zentrale AutoencoderKLFlux2-Struktur

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// VAEConfig holds AutoencoderKLFlux2 configuration
type VAEConfig struct {
	ActFn             string  `json:"act_fn"`                  // "silu"
	BatchNormEps      float32 `json:"batch_norm_eps"`          // 0.0001
	BatchNormMomentum float32 `json:"batch_norm_momentum"`     // 0.1
	BlockOutChannels  []int32 `json:"block_out_channels"`      // [128, 256, 512, 512]
	ForceUpcast       bool    `json:"force_upcast"`            // true
	InChannels        int32   `json:"in_channels"`             // 3
	LatentChannels    int32   `json:"latent_channels"`         // 32
	LayersPerBlock    int32   `json:"layers_per_block"`        // 2
	MidBlockAddAttn   bool    `json:"mid_block_add_attention"` // true
	NormNumGroups     int32   `json:"norm_num_groups"`         // 32
	OutChannels       int32   `json:"out_channels"`            // 3
	PatchSize         []int32 `json:"patch_size"`              // [2, 2]
	SampleSize        int32   `json:"sample_size"`             // 1024
	UsePostQuantConv  bool    `json:"use_post_quant_conv"`     // true
	UseQuantConv      bool    `json:"use_quant_conv"`          // true
}

// BatchNorm2D implements 2D batch normalization with running statistics
type BatchNorm2D struct {
	RunningMean *mlx.Array // [C]
	RunningVar  *mlx.Array // [C]
	Weight      *mlx.Array // [C] gamma
	Bias        *mlx.Array // [C] beta
	Eps         float32
	Momentum    float32
}

// Forward applies batch normalization (inference mode - uses running stats)
// Input and output are in NHWC format [B, H, W, C]
func (bn *BatchNorm2D) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[3]

	// Reshape stats for broadcasting [1, 1, 1, C]
	mean := mlx.Reshape(bn.RunningMean, 1, 1, 1, C)
	variance := mlx.Reshape(bn.RunningVar, 1, 1, 1, C)

	// Normalize: (x - mean) / sqrt(var + eps)
	xNorm := mlx.Sub(x, mean)
	xNorm = mlx.Div(xNorm, mlx.Sqrt(mlx.AddScalar(variance, bn.Eps)))

	// Scale and shift (only if affine=True)
	if bn.Weight != nil {
		weight := mlx.Reshape(bn.Weight, 1, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if bn.Bias != nil {
		bias := mlx.Reshape(bn.Bias, 1, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

// Denormalize inverts the batch normalization
// Used when decoding latents
func (bn *BatchNorm2D) Denormalize(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[3]

	// Reshape stats for broadcasting [1, 1, 1, C]
	mean := mlx.Reshape(bn.RunningMean, 1, 1, 1, C)
	variance := mlx.Reshape(bn.RunningVar, 1, 1, 1, C)

	// Inverse: first undo affine, then undo normalization
	// For affine=False: x_denorm = x * sqrt(var + eps) + mean
	if bn.Bias != nil {
		bias := mlx.Reshape(bn.Bias, 1, 1, 1, C)
		x = mlx.Sub(x, bias)
	}
	if bn.Weight != nil {
		weight := mlx.Reshape(bn.Weight, 1, 1, 1, C)
		x = mlx.Div(x, weight)
	}
	x = mlx.Mul(x, mlx.Sqrt(mlx.AddScalar(variance, bn.Eps)))
	x = mlx.Add(x, mean)

	return x
}

// AutoencoderKLFlux2 is the Flux2 VAE with BatchNorm
type AutoencoderKLFlux2 struct {
	Config *VAEConfig

	// Encoder components (for image editing)
	EncoderConvIn  *Conv2D
	EncoderMid     *VAEMidBlock
	EncoderDown    []*DownEncoderBlock2D
	EncoderNormOut *GroupNormLayer
	EncoderConvOut *Conv2D

	// Decoder components
	DecoderConvIn  *Conv2D
	DecoderMid     *VAEMidBlock
	DecoderUp      []*UpDecoderBlock2D
	DecoderNormOut *GroupNormLayer
	DecoderConvOut *Conv2D

	// Quant conv layers
	QuantConv     *Conv2D
	PostQuantConv *Conv2D

	// BatchNorm for latent normalization
	LatentBN *BatchNorm2D

	// Tiling configuration (nil = no tiling)
	Tiling *vae.TilingConfig
}

// DefaultTilingConfig returns reasonable defaults for tiled decoding
// Matches diffusers: tile_latent_min_size=64, tile_overlap_factor=0.25
func DefaultTilingConfig() *vae.TilingConfig {
	return vae.DefaultTilingConfig()
}
