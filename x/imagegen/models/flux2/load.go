//go:build mlx

// load.go - Model-Loading fuer FLUX.2 Klein.
//
// Dieses Modul enthaelt:
// - Load-Funktion fuer das komplette Modell
// - Tokenizer-, TextEncoder-, Transformer- und VAE-Loading
// - LoadPersistent fuer wiederholte Nutzung

package flux2

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/qwen3"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Load loads the FLUX.2 Klein model from ollama blob storage.
func (m *Model) Load(modelName string) error {
	fmt.Printf("Loading FLUX.2 Klein model from manifest: %s...\n", modelName)
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

	// Load tokenizer
	fmt.Print("  Loading tokenizer... ")
	tokData, err := manifest.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

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
	fmt.Println("OK")

	// Load text encoder
	m.TextEncoder = &qwen3.TextEncoder{}
	if err := m.TextEncoder.Load(manifest, "text_encoder/config.json"); err != nil {
		return fmt.Errorf("text encoder: %w", err)
	}

	// Load transformer
	m.Transformer = &Flux2Transformer2DModel{}
	if err := m.Transformer.Load(manifest); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}

	// Load VAE
	m.VAE = &AutoencoderKLFlux2{}
	if err := m.VAE.Load(manifest); err != nil {
		return fmt.Errorf("VAE: %w", err)
	}

	// Evaluate all weights in a single batch (reduces GPU sync overhead)
	fmt.Print("  Evaluating weights... ")
	allWeights := mlx.Collect(m.TextEncoder)
	allWeights = append(allWeights, mlx.Collect(m.Transformer)...)
	allWeights = append(allWeights, mlx.Collect(m.VAE)...)
	mlx.Eval(allWeights...)
	fmt.Println("OK")

	// Load scheduler config
	m.SchedulerConfig = DefaultSchedulerConfig()
	if schedData, err := manifest.ReadConfig("scheduler/scheduler_config.json"); err == nil {
		if err := json.Unmarshal(schedData, m.SchedulerConfig); err != nil {
			fmt.Printf("  Warning: failed to parse scheduler config: %v\n", err)
		}
	}

	mem := mlx.MetalGetActiveMemory()
	fmt.Printf("  Loaded in %.2fs (%.1f GB VRAM)\n", time.Since(start).Seconds(), float64(mem)/(1024*1024*1024))

	return nil
}

// LoadPersistent loads the model and keeps it in memory for repeated use.
func LoadPersistent(modelName string) (*Model, error) {
	m := &Model{}
	if err := m.Load(modelName); err != nil {
		return nil, err
	}
	return m, nil
}
