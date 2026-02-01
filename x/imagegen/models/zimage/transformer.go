//go:build mlx

// Package zimage - Haupt-Transformer-Modell
// Enthält: Transformer-Struktur, Load, Forward und ForwardWithCache

package zimage

import (
	"fmt"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// Transformer is the full Z-Image DiT model
type Transformer struct {
	TEmbed          *TimestepEmbedder   `weight:"t_embedder"`
	XEmbed          *XEmbedder          `weight:"all_x_embedder"`
	CapEmbed        *CapEmbedder        `weight:"cap_embedder"`
	NoiseRefiners   []*TransformerBlock `weight:"noise_refiner"`
	ContextRefiners []*TransformerBlock `weight:"context_refiner"`
	Layers          []*TransformerBlock `weight:"layers"`
	FinalLayer      *FinalLayer         `weight:"all_final_layer.2-1"`
	XPadToken       *mlx.Array          `weight:"x_pad_token"`
	CapPadToken     *mlx.Array          `weight:"cap_pad_token"`
	*TransformerConfig
}

// Load loads the Z-Image transformer from ollama blob storage.
func (m *Transformer) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading transformer... ")

	// Load config from blob
	var cfg TransformerConfig
	if err := manifest.ReadConfigJSON("transformer/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if len(cfg.AllPatchSize) > 0 {
		cfg.PatchSize = cfg.AllPatchSize[0]
	}
	m.TransformerConfig = &cfg
	m.NoiseRefiners = make([]*TransformerBlock, cfg.NRefinerLayers)
	m.ContextRefiners = make([]*TransformerBlock, cfg.NRefinerLayers)
	m.Layers = make([]*TransformerBlock, cfg.NLayers)

	weights, err := imagegen.LoadWeightsFromManifest(manifest, "transformer")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	return m.loadWeights(weights)
}

// loadWeights loads weights from any WeightSource into the model
func (m *Transformer) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("\u2713")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *Transformer) initComputedFields() {
	cfg := m.TransformerConfig
	m.TEmbed.FreqEmbedSize = 256
	m.FinalLayer.OutDim = m.FinalLayer.Output.OutputDim()
	m.CapEmbed.Norm.Eps = 1e-6

	for _, block := range m.NoiseRefiners {
		initTransformerBlock(block, cfg)
	}
	for _, block := range m.ContextRefiners {
		initTransformerBlock(block, cfg)
	}
	for _, block := range m.Layers {
		initTransformerBlock(block, cfg)
	}
}

// FuseAllQKV fuses QKV projections in all attention layers for efficiency.
// This reduces 3 matmuls to 1 per attention layer, providing ~5-10% speedup.
func (m *Transformer) FuseAllQKV() {
	for _, block := range m.NoiseRefiners {
		block.Attention.FuseQKV()
	}
	for _, block := range m.ContextRefiners {
		block.Attention.FuseQKV()
	}
	for _, block := range m.Layers {
		block.Attention.FuseQKV()
	}
}

// Forward runs the Z-Image transformer with precomputed RoPE
func (m *Transformer) Forward(x *mlx.Array, t *mlx.Array, capFeats *mlx.Array, rope *RoPECache) *mlx.Array {
	imgLen := rope.ImgLen

	// Timestep embedding -> [B, 256]
	temb := m.TEmbed.Forward(mlx.MulScalar(t, m.TransformerConfig.TScale))

	// Embed image patches -> [B, L_img, dim]
	x = m.XEmbed.Forward(x)

	// Embed caption features -> [B, L_cap, dim]
	capEmb := m.CapEmbed.Forward(capFeats)

	eps := m.NormEps

	// Noise refiner: refine image patches with modulation
	for _, refiner := range m.NoiseRefiners {
		x = refiner.Forward(x, temb, rope.ImgCos, rope.ImgSin, eps)
	}

	// Context refiner: refine caption (no modulation)
	for _, refiner := range m.ContextRefiners {
		capEmb = refiner.Forward(capEmb, nil, rope.CapCos, rope.CapSin, eps)
	}

	// Concatenate image and caption for joint attention
	unified := mlx.Concatenate([]*mlx.Array{x, capEmb}, 1)

	// Main transformer layers use full unified RoPE
	for _, layer := range m.Layers {
		unified = layer.Forward(unified, temb, rope.UnifiedCos, rope.UnifiedSin, eps)
	}

	// Extract image tokens only
	unifiedShape := unified.Shape()
	B := unifiedShape[0]
	imgOut := mlx.Slice(unified, []int32{0, 0, 0}, []int32{B, imgLen, unifiedShape[2]})

	// Final layer
	return m.FinalLayer.Forward(imgOut, temb)
}

// ForwardWithCache runs the transformer with layer caching for faster inference.
// On refresh steps (step % cacheInterval == 0), all layers are computed and cached.
// On other steps, shallow layers (0 to cacheLayers-1) reuse cached outputs.
func (m *Transformer) ForwardWithCache(
	x *mlx.Array,
	t *mlx.Array,
	capFeats *mlx.Array,
	rope *RoPECache,
	stepCache *cache.StepCache,
	step int,
	cacheInterval int,
) *mlx.Array {
	imgLen := rope.ImgLen
	cacheLayers := stepCache.NumLayers()
	eps := m.NormEps

	// Timestep embedding -> [B, 256]
	temb := m.TEmbed.Forward(mlx.MulScalar(t, m.TransformerConfig.TScale))

	// Embed image patches -> [B, L_img, dim]
	x = m.XEmbed.Forward(x)

	// Context refiners: compute once on step 0, reuse forever
	// (caption embedding doesn't depend on timestep or latents)
	var capEmb *mlx.Array
	if stepCache.GetConstant() != nil {
		capEmb = stepCache.GetConstant()
	} else {
		capEmb = m.CapEmbed.Forward(capFeats)
		for _, refiner := range m.ContextRefiners {
			capEmb = refiner.Forward(capEmb, nil, rope.CapCos, rope.CapSin, eps)
		}
		stepCache.SetConstant(capEmb)
	}

	// Noise refiners: always compute (depend on x which changes each step)
	for _, refiner := range m.NoiseRefiners {
		x = refiner.Forward(x, temb, rope.ImgCos, rope.ImgSin, eps)
	}

	// Concatenate image and caption for joint attention
	unified := mlx.Concatenate([]*mlx.Array{x, capEmb}, 1)

	// Determine if this is a cache refresh step
	refreshCache := stepCache.ShouldRefresh(step, cacheInterval)

	// Main transformer layers with caching
	for i, layer := range m.Layers {
		if i < cacheLayers && !refreshCache && stepCache.Get(i) != nil {
			// Use cached output for shallow layers
			unified = stepCache.Get(i)
		} else {
			// Compute layer
			unified = layer.Forward(unified, temb, rope.UnifiedCos, rope.UnifiedSin, eps)
			// Cache shallow layer outputs on refresh steps
			if i < cacheLayers && refreshCache {
				stepCache.Set(i, unified)
			}
		}
	}

	// Extract image tokens only
	unifiedShape := unified.Shape()
	B := unifiedShape[0]
	imgOut := mlx.Slice(unified, []int32{0, 0, 0}, []int32{B, imgLen, unifiedShape[2]})

	// Final layer
	return m.FinalLayer.Forward(imgOut, temb)
}
