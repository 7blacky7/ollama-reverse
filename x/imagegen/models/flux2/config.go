//go:build mlx

// config.go - Konfigurationsstrukturen fuer FLUX.2 Klein.
//
// Dieses Modul enthaelt:
// - GenerateConfig fuer Generierungsoptionen
// - Model Struktur mit allen Komponenten
// - Konstanten fuer maximale Aufloesung

package flux2

import (
	"image"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/qwen3"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt        string
	Width         int32                      // Image width (default: 1024)
	Height        int32                      // Image height (default: 1024)
	Steps         int                        // Denoising steps (default: 4 for Klein)
	GuidanceScale float32                    // Guidance scale (default: 1.0, Klein doesn't need CFG)
	Seed          int64                      // Random seed
	Progress      func(step, totalSteps int) // Optional progress callback
	CapturePath   string                     // GPU capture path (debug)
	InputImages   []image.Image              // Reference images for image conditioning (already loaded)
}

// Model represents a FLUX.2 Klein model.
type Model struct {
	ModelName       string
	Tokenizer       *tokenizer.Tokenizer
	TextEncoder     *qwen3.TextEncoder
	Transformer     *Flux2Transformer2DModel
	VAE             *AutoencoderKLFlux2
	SchedulerConfig *SchedulerConfig
}

// TextEncoderLayerIndices are the layers from which to extract text embeddings.
// Diffusers uses hidden_states[9, 18, 27]. In Python, hidden_states[0] is the embedding
// output before any layers, so hidden_states[9] = after layer 8 (0-indexed).
// Go's ForwardWithLayerOutputs captures after layer i runs, so we use [8, 17, 26].
var TextEncoderLayerIndices = []int{8, 17, 26}

// MaxOutputPixels is the maximum output resolution (4 megapixels, ~2048x2048)
const MaxOutputPixels = 2048 * 2048

// MaxRefPixels is the maximum resolution for reference images (smaller to reduce attention memory)
const MaxRefPixels = 728 * 728

// ImageRefScale is the time coordinate offset between reference images (matches diffusers scale=10)
const ImageRefScale = 10

// ImageCondTokens holds encoded reference image tokens.
type ImageCondTokens struct {
	Tokens *mlx.Array // [1, total_tokens, C] - concatenated reference tokens
}
