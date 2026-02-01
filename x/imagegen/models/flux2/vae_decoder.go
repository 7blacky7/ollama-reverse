//go:build mlx

// vae_decoder.go - Decoder-Funktionalitaet des Flux2 VAE
// Enthält Latent-Dekodierung und Tiling-Unterstuetzung

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// denormalizePatchified applies inverse batch normalization to patchified latents.
// Input: [B, L, 128] where 128 = 32 latent channels * 4 (2x2 patch)
// Output: [B, L, 128] denormalized
func (v *AutoencoderKLFlux2) denormalizePatchified(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[2] // 128

	// Reshape stats for broadcasting [1, 1, C]
	mean := mlx.Reshape(v.LatentBN.RunningMean, 1, 1, C)
	variance := mlx.Reshape(v.LatentBN.RunningVar, 1, 1, C)

	// Inverse BN (affine=False): x_denorm = x * sqrt(var + eps) + mean
	if v.LatentBN.Bias != nil {
		bias := mlx.Reshape(v.LatentBN.Bias, 1, 1, C)
		x = mlx.Sub(x, bias)
	}
	if v.LatentBN.Weight != nil {
		weight := mlx.Reshape(v.LatentBN.Weight, 1, 1, C)
		x = mlx.Div(x, weight)
	}
	x = mlx.Mul(x, mlx.Sqrt(mlx.AddScalar(variance, v.LatentBN.Eps)))
	x = mlx.Add(x, mean)

	return x
}

// Decode decodes latent patches to images.
// If Tiling is set, uses tiled decoding to reduce memory for large images.
// latents: [B, L, C*4] patchified latents from transformer
// pH, pW: patch grid dimensions
// Returns: [B, 3, H, W] image tensor
func (v *AutoencoderKLFlux2) Decode(latents *mlx.Array, pH, pW int32) *mlx.Array {
	// Denormalize patchified latents
	z := v.denormalizePatchified(latents)

	// Unpatchify: [B, L, C*4] -> [B, C, H, W]
	z = v.Unpatchify(z, pH, pW, v.Config.LatentChannels)

	// Convert NCHW -> NHWC for processing
	z = mlx.Transpose(z, 0, 2, 3, 1)

	// Use tiled decoding if enabled
	if v.Tiling != nil {
		mlx.Eval(z)
		return vae.DecodeTiled(z, v.Tiling, v.decodeTile)
	}

	// Direct decode (no tiling)
	h := v.decodeTile(z)
	h = mlx.ClipScalar(h, 0.0, 1.0, true, true)
	h = mlx.Transpose(h, 0, 3, 1, 2)
	return h
}

// decodeTile decodes a single latent tile to pixels (internal helper)
// z: [B, H, W, C] latent tile in NHWC format
// Returns: [B, H*8, W*8, 3] pixel tile in NHWC format (before clipping)
func (v *AutoencoderKLFlux2) decodeTile(z *mlx.Array) *mlx.Array {
	// Post-quant conv
	if v.PostQuantConv != nil {
		z = v.PostQuantConv.Forward(z)
	}

	// Decoder
	h := v.DecoderConvIn.Forward(z)
	h = v.DecoderMid.Forward(h)

	for _, upBlock := range v.DecoderUp {
		h = upBlock.Forward(h)
	}

	h = v.DecoderNormOut.Forward(h)
	h = mlx.SiLU(h)
	h = v.DecoderConvOut.Forward(h)

	// VAE outputs [-1, 1], convert to [0, 1]
	h = mlx.MulScalar(h, 0.5)
	h = mlx.AddScalar(h, 0.5)

	return h
}
