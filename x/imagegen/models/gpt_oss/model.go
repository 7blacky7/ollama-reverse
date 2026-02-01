//go:build mlx

// model.go - GPT-OSS Model-Struktur und Laden
// Enthaelt Model-Struktur, Forward-Pass und Load-Funktion.
package gpt_oss

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Model represents the GPT-OSS language model
type Model struct {
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []*Block      `weight:"-"` // loaded manually due to MoE sanitization
	Norm        *nn.RMSNorm   `weight:"model.norm"`
	LMHead      *nn.Linear    `weight:"lm_head"`

	tok *tokenizer.Tokenizer
	*Config
}

// Tokenizer returns the model's tokenizer
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// NumLayers returns the number of transformer layers
func (m *Model) NumLayers() int { return len(m.Layers) }

// VocabSize returns the vocabulary size
func (m *Model) VocabSize() int32 { return m.Config.VocabSize }

// NewCache creates a new KV cache for generation
func (m *Model) NewCache(int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if layer.LayerType == "sliding_attention" && m.SlidingWindow > 0 {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

// Forward performs the model forward pass
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	x := m.EmbedTokens.Forward(tokens)

	// Find representative cache indices for sliding window attention
	var swaIdx int = -1
	for i, layer := range m.Layers {
		if layer.LayerType == "sliding_attention" {
			swaIdx = i
			break
		}
	}

	// Create masks once at model level
	var fullMask, swaMask *mlx.Array
	var fullMaskMode, swaMaskMode string

	if L > 1 {
		fullMaskMode = "causal"
		if swaIdx >= 0 && m.SlidingWindow > 0 && caches != nil {
			c := caches[swaIdx]
			offset := c.Offset()
			windowSize := int(m.SlidingWindow)
			cacheLen := min(int(L), windowSize)
			if offset > 0 {
				cacheLen = min(c.Len()+int(L), windowSize)
			}
			if int(L) > windowSize {
				swaMask = CreateSlidingWindowMask(int(L), offset, offset+int(L)-cacheLen, cacheLen, windowSize)
			} else {
				swaMaskMode = "causal"
			}
		} else {
			swaMaskMode = "causal"
		}
	}

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		mask, maskMode := fullMask, fullMaskMode
		if layer.LayerType == "sliding_attention" {
			mask, maskMode = swaMask, swaMaskMode
		}
		x = layer.Forward(x, c, B, L, mask, maskMode, m.Config)
	}

	return m.LMHead.Forward(m.Norm.Forward(x, m.RMSNormEps))
}

// MaxContextLength returns the maximum context length
func (m *Model) MaxContextLength() int32 {
	if m.RopeScaling != nil && m.RopeScaling.OriginalMaxPositionEmbeddings > 0 {
		return m.RopeScaling.OriginalMaxPositionEmbeddings
	}
	return 131072
}

// Load loads the GPT-OSS model from a directory
func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Block, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	// Load simple weights via struct tags
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Load layers with custom MoE handling
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Block{}
		if err := safetensors.LoadModule(layer, weights, prefix); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}

		// Initialize attention YaRN
		layer.Attention.initYarn(&cfg)

		// Load MoE with weight sanitization
		moe := sanitizeExpertWeights(weights, prefix)
		moe.Router = layer.MLP.Router // Router was loaded by LoadModule
		moe.TopK = cfg.NumExpertsPerTok
		moe.HiddenSize = cfg.HiddenSize
		layer.MLP = moe

		// Set layer type
		layer.LayerType = "full_attention"
		if int(i) < len(cfg.LayerTypes) {
			layer.LayerType = cfg.LayerTypes[i]
		}

		m.Layers[i] = layer
	}

	// Release safetensors BEFORE eval - lazy arrays have captured data,
	// this reduces peak memory by freeing mmap during materialization
	weights.ReleaseAll()
	mlx.Eval(mlx.Collect(m)...)

	return m, nil
}
