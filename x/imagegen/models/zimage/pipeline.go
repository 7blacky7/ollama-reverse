//go:build mlx

// pipeline.go - Interne Denoising-Pipeline fuer Z-Image
// Enthaelt die Hauptlogik fuer den Diffusionsprozess inkl. TeaCache und CFG.
package zimage

import (
	"context"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// generate is the internal denoising pipeline.
func (m *Model) generate(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	// Apply defaults
	cfg.applyDefaults()

	// Enable fused QKV if requested (only fuse once)
	if cfg.FusedQKV && !m.qkvFused {
		m.Transformer.FuseAllQKV()
		m.qkvFused = true
		fmt.Println("  Fused QKV enabled")
	}

	useCFG := cfg.NegativePrompt != ""
	tcfg := m.Transformer.TransformerConfig
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	hTok := latentH / tcfg.PatchSize
	wTok := latentW / tcfg.PatchSize

	// Text encoding with padding to multiple of 32
	posEmb, negEmb := m.encodePrompts(cfg, useCFG)

	// Scheduler
	scheduler := NewFlowMatchEulerScheduler(DefaultFlowMatchSchedulerConfig())
	scheduler.SetTimestepsWithMu(cfg.Steps, CalculateShift(hTok*wTok))

	// Init latents [B, C, H, W]
	latents := scheduler.InitNoise([]int32{1, tcfg.InChannels, latentH, latentW}, cfg.Seed)
	mlx.Eval(latents)

	// RoPE cache
	ropeCache := m.Transformer.PrepareRoPECache(hTok, wTok, posEmb.Shape()[1])
	mlx.Keep(ropeCache.ImgCos, ropeCache.ImgSin, ropeCache.CapCos, ropeCache.CapSin,
		ropeCache.UnifiedCos, ropeCache.UnifiedSin)
	mlx.Eval(ropeCache.UnifiedCos)

	// Pre-compute batched embeddings for CFG (outside the loop for efficiency)
	var batchedEmb *mlx.Array
	if useCFG {
		batchedEmb = mlx.Concatenate([]*mlx.Array{posEmb, negEmb}, 0)
		mlx.Keep(batchedEmb)
		mlx.Eval(batchedEmb)
	}

	// TeaCache for timestep-aware caching
	teaCache := m.initTeaCache(cfg, useCFG)

	// cleanup frees all kept arrays when we need to abort early
	cleanup := func() {
		posEmb.Free()
		if negEmb != nil {
			negEmb.Free()
		}
		ropeCache.ImgCos.Free()
		ropeCache.ImgSin.Free()
		ropeCache.CapCos.Free()
		ropeCache.CapSin.Free()
		ropeCache.UnifiedCos.Free()
		ropeCache.UnifiedSin.Free()
		if batchedEmb != nil {
			batchedEmb.Free()
		}
		if teaCache != nil {
			teaCache.Free()
		}
		latents.Free()
	}

	// Denoising loop
	if cfg.Progress != nil {
		cfg.Progress(0, cfg.Steps)
	}
	for i := 0; i < cfg.Steps; i++ {
		// Check for cancellation
		if ctx != nil {
			select {
			case <-ctx.Done():
				cleanup()
				return nil, ctx.Err()
			default:
			}
		}
		stepStart := time.Now()

		// GPU capture on step 2 if requested
		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStartCapture(cfg.CapturePath)
		}

		tCurr := scheduler.Timesteps[i]
		noisePred := m.computeNoisePrediction(cfg, useCFG, i, tCurr, latents, posEmb, batchedEmb, ropeCache, teaCache, latentH, latentW, tcfg)

		oldLatents := latents
		latents = scheduler.Step(noisePred, latents, i)

		mlx.Eval(latents)
		oldLatents.Free()

		if cfg.CapturePath != "" && i == 1 {
			mlx.MetalStopCapture()
		}

		activeMem := float64(mlx.MetalGetActiveMemory()) / (1024 * 1024 * 1024)
		peakMem := float64(mlx.MetalGetPeakMemory()) / (1024 * 1024 * 1024)
		fmt.Printf("  Step %d/%d: t=%.4f (%.2fs) [%.1f GB active, %.1f GB peak]\n",
			i+1, cfg.Steps, tCurr, time.Since(stepStart).Seconds(), activeMem, peakMem)

		if cfg.Progress != nil {
			cfg.Progress(i+1, cfg.Steps)
		}
	}

	// Free denoising temporaries before VAE decode
	posEmb.Free()
	if negEmb != nil {
		negEmb.Free()
	}
	ropeCache.ImgCos.Free()
	ropeCache.ImgSin.Free()
	ropeCache.CapCos.Free()
	ropeCache.CapSin.Free()
	ropeCache.UnifiedCos.Free()
	ropeCache.UnifiedSin.Free()
	if batchedEmb != nil {
		batchedEmb.Free()
	}
	if teaCache != nil {
		hits, misses := teaCache.Stats()
		fmt.Printf("  TeaCache stats: %d hits, %d misses (%.1f%% cache rate)\n",
			hits, misses, float64(hits)/float64(hits+misses)*100)
		teaCache.Free()
	}

	// VAE decode - enable tiling for larger images to reduce memory
	if latentH > 64 || latentW > 64 {
		m.VAEDecoder.Tiling = vae.DefaultTilingConfig()
	}
	decoded := m.VAEDecoder.Decode(latents)
	latents.Free()

	return decoded, nil
}

// encodePrompts encodes the prompt(s) and pads to multiple of 32.
func (m *Model) encodePrompts(cfg *GenerateConfig, useCFG bool) (*mlx.Array, *mlx.Array) {
	posEmb, _ := m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.Prompt, 512, false)
	var negEmb *mlx.Array
	if useCFG {
		negEmb, _ = m.TextEncoder.EncodePrompt(m.Tokenizer, cfg.NegativePrompt, 512, false)
	}

	// Pad both to same length (multiple of 32)
	maxLen := posEmb.Shape()[1]
	if useCFG && negEmb.Shape()[1] > maxLen {
		maxLen = negEmb.Shape()[1]
	}
	if pad := (32 - (maxLen % 32)) % 32; pad > 0 {
		maxLen += pad
	}

	posEmb = padToLength(posEmb, maxLen)
	if useCFG {
		negEmb = padToLength(negEmb, maxLen)
		mlx.Keep(posEmb, negEmb)
		mlx.Eval(posEmb, negEmb)
	} else {
		mlx.Keep(posEmb)
		mlx.Eval(posEmb)
	}

	return posEmb, negEmb
}

// initTeaCache initializes TeaCache if enabled.
func (m *Model) initTeaCache(cfg *GenerateConfig, useCFG bool) *cache.TeaCache {
	if !cfg.TeaCache {
		return nil
	}
	skipEarly := 0
	if useCFG {
		skipEarly = 3 // Skip first 3 steps for CFG to preserve structure
	}
	teaCache := cache.NewTeaCache(&cache.TeaCacheConfig{
		Threshold:      cfg.TeaCacheThreshold,
		RescaleFactor:  1.0,
		SkipEarlySteps: skipEarly,
	})
	if useCFG {
		fmt.Printf("  TeaCache enabled (CFG mode): threshold=%.2f, skip first %d steps\n", cfg.TeaCacheThreshold, skipEarly)
	} else {
		fmt.Printf("  TeaCache enabled: threshold=%.2f\n", cfg.TeaCacheThreshold)
	}
	return teaCache
}

// computeNoisePrediction computes the noise prediction for a single step.
func (m *Model) computeNoisePrediction(cfg *GenerateConfig, useCFG bool, step int, tCurr float32,
	latents, posEmb, batchedEmb *mlx.Array, ropeCache *RoPECache, teaCache *cache.TeaCache,
	latentH, latentW int32, tcfg *TransformerConfig) *mlx.Array {

	shouldCompute := teaCache == nil || teaCache.ShouldCompute(step, tCurr)

	if shouldCompute {
		timestep := mlx.ToBFloat16(mlx.NewArray([]float32{1.0 - tCurr}, []int32{1}))
		patches := PatchifyLatents(latents, tcfg.PatchSize)

		if useCFG {
			return m.computeCFGPrediction(cfg, patches, timestep, batchedEmb, ropeCache, teaCache, tCurr, latentH, latentW, tcfg)
		}
		// Non-CFG forward pass
		output := m.Transformer.Forward(patches, timestep, posEmb, ropeCache)
		noisePred := UnpatchifyLatents(output, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
		noisePred = mlx.Neg(noisePred)

		if teaCache != nil {
			teaCache.UpdateCache(noisePred, tCurr)
			mlx.Keep(teaCache.Arrays()...)
		}
		return noisePred
	}

	// Use cached prediction
	if useCFG && teaCache != nil && teaCache.HasCFGCache() {
		posPred, negPred := teaCache.GetCFGCached()
		diff := mlx.Sub(posPred, negPred)
		scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
		fmt.Printf("    [TeaCache: reusing cached pos/neg outputs]\n")
		return mlx.Add(negPred, scaledDiff)
	}
	fmt.Printf("    [TeaCache: reusing cached output]\n")
	return teaCache.GetCached()
}

// computeCFGPrediction computes CFG noise prediction with batched forward pass.
func (m *Model) computeCFGPrediction(cfg *GenerateConfig, patches, timestep, batchedEmb *mlx.Array,
	ropeCache *RoPECache, teaCache *cache.TeaCache, tCurr float32,
	latentH, latentW int32, tcfg *TransformerConfig) *mlx.Array {

	// CFG Batching: single forward pass with batch=2
	batchedPatches := mlx.Tile(patches, []int32{2, 1, 1})
	batchedTimestep := mlx.Tile(timestep, []int32{2})

	batchedOutput := m.Transformer.Forward(batchedPatches, batchedTimestep, batchedEmb, ropeCache)

	// Split output: [2, L, D] -> pos [1, L, D], neg [1, L, D]
	outputShape := batchedOutput.Shape()
	L := outputShape[1]
	D := outputShape[2]
	posOutput := mlx.Slice(batchedOutput, []int32{0, 0, 0}, []int32{1, L, D})
	negOutput := mlx.Slice(batchedOutput, []int32{1, 0, 0}, []int32{2, L, D})

	// Convert to noise predictions
	posPred := UnpatchifyLatents(posOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
	posPred = mlx.Neg(posPred)
	negPred := UnpatchifyLatents(negOutput, tcfg.PatchSize, latentH, latentW, tcfg.InChannels)
	negPred = mlx.Neg(negPred)

	// Cache pos/neg separately for TeaCache
	if teaCache != nil {
		teaCache.UpdateCFGCache(posPred, negPred, tCurr)
		mlx.Keep(teaCache.Arrays()...)
	}

	// Apply CFG: noisePred = neg + scale * (pos - neg)
	diff := mlx.Sub(posPred, negPred)
	scaledDiff := mlx.MulScalar(diff, cfg.CFGScale)
	return mlx.Add(negPred, scaledDiff)
}
