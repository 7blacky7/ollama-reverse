//go:build mlx

// vae_encoder.go - Encoder-Funktionalitaet des Flux2 VAE
// Enthält Bildkodierung und Patchify-Operationen

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Patchify converts latents [B, C, H, W] to patches [B, H*W/4, C*4] using 2x2 patches
// This is the inverse of the VAE's patchify for feeding to transformer
func (vae *AutoencoderKLFlux2) Patchify(latents *mlx.Array) *mlx.Array {
	shape := latents.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]

	patchH := vae.Config.PatchSize[0]
	patchW := vae.Config.PatchSize[1]

	pH := H / patchH
	pW := W / patchW

	// [B, C, H, W] -> [B, C, pH, patchH, pW, patchW]
	x := mlx.Reshape(latents, B, C, pH, patchH, pW, patchW)
	// [B, C, pH, patchH, pW, patchW] -> [B, pH, pW, C, patchH, patchW]
	x = mlx.Transpose(x, 0, 2, 4, 1, 3, 5)
	// [B, pH, pW, C, patchH, patchW] -> [B, pH*pW, C*patchH*patchW]
	return mlx.Reshape(x, B, pH*pW, C*patchH*patchW)
}

// Unpatchify converts patches [B, L, C*4] back to [B, C, H, W]
func (vae *AutoencoderKLFlux2) Unpatchify(patches *mlx.Array, pH, pW, C int32) *mlx.Array {
	shape := patches.Shape()
	B := shape[0]

	patchH := vae.Config.PatchSize[0]
	patchW := vae.Config.PatchSize[1]

	// [B, pH*pW, C*patchH*patchW] -> [B, pH, pW, C, patchH, patchW]
	x := mlx.Reshape(patches, B, pH, pW, C, patchH, patchW)
	// [B, pH, pW, C, patchH, patchW] -> [B, C, pH, patchH, pW, patchW]
	x = mlx.Transpose(x, 0, 3, 1, 4, 2, 5)
	// [B, C, pH, patchH, pW, patchW] -> [B, C, H, W]
	H := pH * patchH
	W := pW * patchW
	return mlx.Reshape(x, B, C, H, W)
}

// EncodeImage encodes an image to normalized latents.
// image: [B, 3, H, W] image tensor in [-1, 1]
// Returns: [B, L, C*4] patchified normalized latents
func (vae *AutoencoderKLFlux2) EncodeImage(image *mlx.Array) *mlx.Array {
	// Convert NCHW -> NHWC
	x := mlx.Transpose(image, 0, 2, 3, 1)

	// Encoder
	h := vae.EncoderConvIn.Forward(x)

	for _, downBlock := range vae.EncoderDown {
		h = downBlock.Forward(h)
	}

	h = vae.EncoderMid.Forward(h)
	h = vae.EncoderNormOut.Forward(h)
	h = mlx.SiLU(h)
	h = vae.EncoderConvOut.Forward(h)

	// Quant conv outputs [B, H, W, 2*latent_channels] (mean + logvar)
	if vae.QuantConv != nil {
		h = vae.QuantConv.Forward(h)
	}

	// Take only the mean (first latent_channels) - deterministic encoding
	// h is [B, H, W, 64] -> take first 32 channels for mean
	shape := h.Shape()
	latentChannels := vae.Config.LatentChannels // 32
	h = mlx.Slice(h, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], shape[2], latentChannels})

	// Convert NHWC -> NCHW for patchifying
	h = mlx.Transpose(h, 0, 3, 1, 2)

	// Patchify: [B, C, H, W] -> [B, L, C*4]
	h = vae.Patchify(h)

	// Apply BatchNorm on patchified latents [B, L, 128]
	// The BatchNorm has 128 channels matching the patchified dimension
	h = vae.normalizePatchified(h)

	return h
}

// normalizePatchified applies batch normalization to patchified latents.
// Input: [B, L, 128] where 128 = 32 latent channels * 4 (2x2 patch)
// Output: [B, L, 128] normalized
func (vae *AutoencoderKLFlux2) normalizePatchified(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	C := shape[2] // 128

	// Reshape stats for broadcasting [1, 1, C]
	mean := mlx.Reshape(vae.LatentBN.RunningMean, 1, 1, C)
	variance := mlx.Reshape(vae.LatentBN.RunningVar, 1, 1, C)

	// Normalize: (x - mean) / sqrt(var + eps)
	xNorm := mlx.Sub(x, mean)
	xNorm = mlx.Div(xNorm, mlx.Sqrt(mlx.AddScalar(variance, vae.LatentBN.Eps)))

	// Scale and shift (only if affine=True)
	if vae.LatentBN.Weight != nil {
		weight := mlx.Reshape(vae.LatentBN.Weight, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if vae.LatentBN.Bias != nil {
		bias := mlx.Reshape(vae.LatentBN.Bias, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}
