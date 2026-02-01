//go:build mlx

// model.go - Z-Image Modell-Struktur und Laden
// Enthaelt die Model-Struktur und die Load-Methode zum Laden aus Ollama Blob Storage.
package zimage

import (
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Model represents a Z-Image diffusion model.
type Model struct {
	ModelName   string
	Tokenizer   *tokenizer.Tokenizer
	TextEncoder *Qwen3TextEncoder
	Transformer *Transformer
	VAEDecoder  *VAEDecoder
	qkvFused    bool // Track if QKV has been fused (do only once)
}

// Load loads the Z-Image model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading Z-Image model from manifest: %s...\n", modelName)
	start := time.Now()

	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
		mlx.EnableCompile()
	}

	m.ModelName = modelName

	// Load manifest
	manifest, err := imagegen.LoadManifest(modelName)
	if err != nil {
		return fmt.Errorf("load manifest: %w", err)
	}

	// Load tokenizer from manifest with config
	fmt.Print("  Loading tokenizer... ")
	tokData, err := manifest.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	// Try to read tokenizer config files from manifest
	tokConfig := &tokenizer.TokenizerConfig{}
	if data, err := manifest.ReadConfig("tokenizer/tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = data
	}
	if data, err := manifest.ReadConfig("tokenizer/generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = data
	}
	if data, err := manifest.ReadConfig("tokenizer/special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = data
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	m.Tokenizer = tok
	fmt.Println("done")

	// Load text encoder
	m.TextEncoder = &Qwen3TextEncoder{}
	if err := m.TextEncoder.Load(manifest, "text_encoder/config.json"); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.TextEncoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load transformer
	m.Transformer = &Transformer{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}
	mlx.Eval(mlx.Collect(m.Transformer)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	// Load VAE decoder
	m.VAEDecoder = &VAEDecoder{}
	if err := m.VAEDecoder.Load(manifest); err != nil {
		return fmt.Errorf("VAE decoder: %w", err)
	}
	mlx.Eval(mlx.Collect(m.VAEDecoder)...)
	fmt.Printf("  (%.1f GB, peak %.1f GB)\n",
		float64(mlx.MetalGetActiveMemory())/(1024*1024*1024),
		float64(mlx.MetalGetPeakMemory())/(1024*1024*1024))

	mem := mlx.MetalGetActiveMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

	return nil
}
