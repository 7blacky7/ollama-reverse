//go:build mlx

// Modul: encoder.go
// Beschreibung: Hauptstruktur des Qwen3 Text-Encoders.
// Enthält: TextEncoder-Struct, Load, loadWeights, initComputedFields, Forward, ForwardWithLayerOutputs.

package qwen3

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// TextEncoder is the full Qwen3 encoder
type TextEncoder struct {
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []*Block      `weight:"model.layers"`
	FinalNorm   *nn.RMSNorm   `weight:"model.norm"`
	*Config
}

// Load loads the Qwen3 text encoder from ollama blob storage.
func (m *TextEncoder) Load(manifest *imagegen.ModelManifest, configPath string) error {
	fmt.Print("  Loading text encoder... ")

	// Load config from blob
	var cfg Config
	if err := manifest.ReadConfigJSON(configPath, &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg
	m.Layers = make([]*Block, cfg.NumHiddenLayers)

	// Load weights from tensor blobs
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "text_encoder")
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
func (m *TextEncoder) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("done")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *TextEncoder) initComputedFields() {
	cfg := m.Config
	m.FinalNorm.Eps = cfg.RMSNormEps
	for _, block := range m.Layers {
		// Attention
		block.Attention.NHeads = cfg.NumAttentionHeads
		block.Attention.NKVHeads = cfg.NumKeyValueHeads
		block.Attention.HeadDim = cfg.HeadDim
		block.Attention.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
		block.Attention.RopeTheta = cfg.RopeTheta
		block.Attention.QNorm.Eps = cfg.RMSNormEps
		block.Attention.KNorm.Eps = cfg.RMSNormEps
		// Block norms
		block.InputLayerNorm.Eps = cfg.RMSNormEps
		block.PostAttnLayerNorm.Eps = cfg.RMSNormEps
	}
}

// Forward encodes text tokens with provided attention mask (LxL) and mask mode.
func (te *TextEncoder) Forward(tokens *mlx.Array, attnMask *mlx.Array, maskMode string) *mlx.Array {
	h := te.EmbedTokens.Forward(tokens)
	eps := te.RMSNormEps

	for _, layer := range te.Layers {
		h = layer.Forward(h, eps, attnMask, maskMode)
	}

	// Apply final RMS norm
	h = te.FinalNorm.Forward(h, eps)

	return h
}

// ForwardWithLayerOutputs encodes text tokens and returns hidden states from specified layers.
// This is used by Flux2 which needs embeddings from specific intermediate layers.
func (te *TextEncoder) ForwardWithLayerOutputs(tokens *mlx.Array, layerIndices []int, attnMask *mlx.Array, maskMode string) []*mlx.Array {
	h := te.EmbedTokens.Forward(tokens)
	eps := te.RMSNormEps

	outputs := make([]*mlx.Array, len(layerIndices))
	layerSet := make(map[int]int)
	for i, idx := range layerIndices {
		layerSet[idx] = i
	}

	for i, layer := range te.Layers {
		h = layer.Forward(h, eps, attnMask, maskMode)
		if outIdx, ok := layerSet[i]; ok {
			outputs[outIdx] = h
		}
	}

	return outputs
}
