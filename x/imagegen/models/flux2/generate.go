//go:build mlx

// generate.go - Bildgenerierungs-API fuer FLUX.2 Klein.
//
// Dieses Modul enthaelt:
// - Generate und GenerateWithProgress Funktionen
// - GenerateFromConfig Haupteinstiegspunkt
// - GenerateImage/GenerateImageWithInputs Interfaces

package flux2

import (
	"context"
	"fmt"
	"image"
	"time"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Generate creates an image from a prompt.
func (m *Model) Generate(prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// GenerateWithProgress creates an image with progress callback.
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateFromConfig generates an image using the unified config struct.
func (m *Model) GenerateFromConfig(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(ctx, cfg)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Generated in %.2fs (%d steps)\n", time.Since(start).Seconds(), cfg.Steps)
	return result, nil
}

// GenerateImage implements runner.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(ctx, &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateImageWithInputs implements runner.ImageEditModel interface.
// It generates an image conditioned on the provided input images for image editing.
func (m *Model) GenerateImageWithInputs(ctx context.Context, prompt string, width, height int32, steps int, seed int64, inputImages []image.Image, progress func(step, total int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(ctx, &GenerateConfig{
		Prompt:      prompt,
		Width:       width,
		Height:      height,
		Steps:       steps,
		Seed:        seed,
		InputImages: inputImages,
		Progress:    progress,
	})
}

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Enable MLX compilation for fused kernels
	mlx.EnableCompile()

	// Apply defaults
	if cfg.Steps <= 0 {
		cfg.Steps = 4 // Klein default: 4 steps for distilled model
	}
	if cfg.GuidanceScale <= 0 {
		cfg.GuidanceScale = 1.0 // Klein doesn't need guidance
	}

	// Determine output dimensions
	cfg.Width, cfg.Height = computeOutputDimensions(cfg)
	fmt.Printf("  Output: %dx%d\n", cfg.Width, cfg.Height)

	tcfg := m.Transformer.TransformerConfig
	patchSize := m.VAE.Config.PatchSize

	// Latent dimensions: image / 8 (VAE downscale) / patch_size
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	patchH := latentH / patchSize[0]
	patchW := latentW / patchSize[1]
	imgSeqLen := patchH * patchW

	// Text encoding with multi-layer extraction (no padding, use true sequence length)
	fmt.Print("  Encoding prompt... ")
	promptEmbeds, textLen := m.TextEncoder.EncodePromptWithLayers(m.Tokenizer, cfg.Prompt, 512, TextEncoderLayerIndices, false)
	fmt.Println("OK")

	// Encode reference images if provided
	var refTokens *ImageCondTokens
	var refHeights, refWidths []int32
	if len(cfg.InputImages) > 0 {
		fmt.Printf("  Encoding %d reference image(s):\n", len(cfg.InputImages))

		var err error
		refTokens, err = m.EncodeImageRefs(cfg.InputImages)
		if err != nil {
			return nil, fmt.Errorf("encode reference images: %w", err)
		}

		// Extract heights/widths for RoPE computation (same limits as EncodeImageRefs)
		limitPixels := MaxRefPixels
		if len(cfg.InputImages) > 1 {
			limitPixels = MaxRefPixels / 2
		}
		for _, img := range cfg.InputImages {
			_, w, h := PrepareImage(img, limitPixels)
			refHeights = append(refHeights, int32(h/16))
			refWidths = append(refWidths, int32(w/16))
		}
	}

	// Scheduler
	scheduler := NewFlowMatchScheduler(m.SchedulerConfig)
	scheduler.SetTimestepsWithMu(cfg.Steps, CalculateShift(imgSeqLen, cfg.Steps))

	// Init latents in packed form [B, C*4, H/2, W/2] like diffusers
	latentChannels := m.VAE.Config.LatentChannels
	packedChannels := latentChannels * 4 // 32 * 4 = 128
	latents := scheduler.InitNoise([]int32{1, packedChannels, patchH, patchW}, cfg.Seed)

	// Pack latents (transpose): [B, C, H, W] -> [B, H*W, C]
	patches := packLatents(latents)
	noiseSeqLen := patches.Shape()[1]

	// RoPE cache - includes reference images if present
	rope := PrepareRoPECache(textLen, patchH, patchW, tcfg.AxesDimsRoPE, tcfg.RopeTheta, refHeights, refWidths, ImageRefScale)

	// Cleanup setup arrays when done
	defer func() {
		rope.Cos.Free()
		rope.Sin.Free()
		promptEmbeds.Free()
		if refTokens != nil {
			refTokens.Tokens.Free()
		}
	}()

	// Pre-compute all timesteps before the loop
	timesteps := make([]*mlx.Array, cfg.Steps)
	for i := 0; i < cfg.Steps; i++ {
		tCurr := scheduler.Timesteps[i] / float32(m.SchedulerConfig.NumTrainTimesteps)
		timesteps[i] = mlx.ToBFloat16(mlx.NewArray([]float32{tCurr}, []int32{1}))
	}

	// Evaluate setup arrays
	fmt.Print("  Evaluating setup... ")
	setupStart := time.Now()
	toEval := []*mlx.Array{promptEmbeds, patches, rope.Cos, rope.Sin}
	toEval = append(toEval, timesteps...)
	if refTokens != nil {
		toEval = append(toEval, refTokens.Tokens)
	}
	mlx.Eval(toEval...)
	mlx.MetalResetPeakMemory()
	fmt.Printf("OK (%.2fs, %.1f GB)\n", time.Since(setupStart).Seconds(),
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024))

	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps)
	}

	// Run denoising loop
	patches = m.runDenoisingLoop(ctx, cfg, patches, promptEmbeds, timesteps, rope, refTokens, noiseSeqLen, scheduler)
	if patches == nil {
		return nil, ctx.Err()
	}

	// Free timesteps
	for _, ts := range timesteps {
		ts.Free()
	}

	// VAE decode
	decoded := m.decodeLatents(patches, patchH, patchW)

	// Free patches
	patches.Free()

	return decoded, nil
}
