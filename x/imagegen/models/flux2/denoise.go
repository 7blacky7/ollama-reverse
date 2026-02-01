//go:build mlx

// denoise.go - Denoising-Loop und VAE-Decoding fuer FLUX.2.
//
// Dieses Modul enthaelt:
// - runDenoisingLoop fuer die Denoising-Schritte
// - decodeLatents fuer VAE-Decoding
// - computeOutputDimensions fuer Dimensionsberechnung
// - packLatents fuer Tensor-Umformung

package flux2

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// runDenoisingLoop executes the denoising steps.
func (m *Model) runDenoisingLoop(ctx context.Context, cfg *GenerateConfig, patches, promptEmbeds *mlx.Array, timesteps []*mlx.Array, rope *RoPECache, refTokens *ImageCondTokens, noiseSeqLen int32, scheduler *FlowMatchScheduler) *mlx.Array {
	loopStart := time.Now()
	stepStart := time.Now()

	for i := 0; i < cfg.Steps; i++ {
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil
			default:
			}
		}

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		timestep := timesteps[i]

		// Prepare input - concatenate noise patches with reference tokens if present
		imgInput := patches
		if refTokens != nil {
			imgInput = mlx.Concatenate([]*mlx.Array{patches, refTokens.Tokens}, 1)
		}

		// Transformer forward pass
		output := m.Transformer.Forward(imgInput, promptEmbeds, timestep, rope)

		// If we concatenated reference tokens, slice to only get noise portion
		if refTokens != nil {
			output = mlx.Slice(output, []int32{0, 0, 0}, []int32{1, noiseSeqLen, output.Shape()[2]})
		}

		// Scheduler step
		newPatches := scheduler.Step(output, patches, i)

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		mlx.Eval(newPatches)
		patches = newPatches

		elapsed := time.Since(stepStart).Seconds()
		peakGB := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		if i == 0 {
			fmt.Printf("    step %d: %.2fs (JIT warmup), peak %.1f GB\n", i+1, elapsed, peakGB)
		} else {
			fmt.Printf("    step %d: %.2fs, peak %.1f GB\n", i+1, elapsed, peakGB)
		}
		stepStart = time.Now()
		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}
	}

	loopTime := time.Since(loopStart).Seconds()
	peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
	fmt.Printf("  Denoised %d steps in %.2fs (%.2fs/step), peak %.1f GB\n",
		cfg.Steps, loopTime, loopTime/float64(cfg.Steps), peakMem)

	return patches
}

// decodeLatents decodes latent patches using the VAE.
func (m *Model) decodeLatents(patches *mlx.Array, patchH, patchW int32) *mlx.Array {
	fmt.Print("  Decoding VAE... ")
	vaeStart := time.Now()

	// Enable tiling for images > 512x512 (latent > 64x64)
	if patchH*2 > 64 || patchW*2 > 64 {
		m.VAE.Tiling = DefaultTilingConfig()
	}
	decoded := m.VAE.Decode(patches, patchH, patchW)
	mlx.Eval(decoded)

	fmt.Printf("OK (%.2fs, peak %.1f GB)\n", time.Since(vaeStart).Seconds(),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	return decoded
}

// computeOutputDimensions calculates final output dimensions from config.
func computeOutputDimensions(cfg *GenerateConfig) (int32, int32) {
	width, height := cfg.Width, cfg.Height

	// With input images, compute missing dimension from aspect ratio
	if len(cfg.InputImages) > 0 {
		bounds := cfg.InputImages[0].Bounds()
		imgW, imgH := bounds.Dx(), bounds.Dy()
		aspectRatio := float64(imgH) / float64(imgW)
		if width > 0 && height <= 0 {
			height = int32(math.Round(float64(width)*aspectRatio/16) * 16)
		} else if height > 0 && width <= 0 {
			width = int32(math.Round(float64(height)/aspectRatio/16) * 16)
		} else if width <= 0 && height <= 0 {
			width = int32(imgW)
			height = int32(imgH)
		}
	}
	if width <= 0 {
		width = 1024
	}
	if height <= 0 {
		height = 1024
	}

	// Cap to max pixels, preserve aspect ratio, round to multiple of 16
	pixels := int(width) * int(height)
	if pixels > MaxOutputPixels {
		scale := math.Sqrt(float64(MaxOutputPixels) / float64(pixels))
		width = int32(math.Round(float64(width) * scale / 16) * 16)
		height = int32(math.Round(float64(height) * scale / 16) * 16)
	}
	height = int32((height + 8) / 16 * 16) // round to nearest 16
	width = int32((width + 8) / 16 * 16)

	return width, height
}

// packLatents converts [B, C, H, W] to [B, H*W, C] (matches diffusers _pack_latents)
func packLatents(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	C := shape[1]
	H := shape[2]
	W := shape[3]
	// [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
	x = mlx.Reshape(x, B, C, H*W)
	return mlx.Transpose(x, 0, 2, 1)
}
