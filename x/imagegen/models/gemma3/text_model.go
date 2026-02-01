//go:build mlx

// text_model.go - Gemma 3 Text-Modell Implementation.
//
// Dieses Modul enthaelt:
// - TextModel Struktur und Loading
// - Forward-Pass fuer das Text-Modell
// - Layer-Forward-Implementierung
// - Cache-Verwaltung

package gemma3

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

// TextModel is the Gemma 3 text-only model
type TextModel struct {
	EmbedTokens *nn.Embedding   `weight:"model.embed_tokens"`
	Layers      []*DecoderLayer `weight:"model.layers"`
	Norm        *nn.RMSNorm     `weight:"model.norm"`
	Output      *nn.Linear      `weight:"-"` // Tied to EmbedTokens, set manually

	// Precomputed (1 + weight) for Gemma-style RMSNorm to avoid allocation per forward
	NormScaled *mlx.Array `weight:"-"`

	tok *tokenizer.Tokenizer
	*TextConfig
}

// LoadText loads the text-only Gemma 3 model
func LoadText(modelPath string) (*TextModel, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var cfg TextConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Compute scale
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Set defaults if not specified
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RopeLocalBaseFreq == 0 {
		cfg.RopeLocalBaseFreq = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &TextModel{
		Layers:     make([]*DecoderLayer, cfg.NumHiddenLayers),
		TextConfig: &cfg,
		tok:        tok,
	}

	// Initialize layer metadata
	for i := range m.Layers {
		m.Layers[i] = &DecoderLayer{
			LayerIdx:  int32(i),
			IsSliding: isLayerSliding(int32(i), cfg.SlidingWindowPattern),
		}
	}

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Tied embeddings for output
	m.Output = nn.NewLinear(m.EmbedTokens.Weight, nil)

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	// Precompute (1 + weight) for Gemma-style RMSNorm to avoid per-forward allocation
	precomputeGemmaScaledWeights(m)

	return m, nil
}

// precomputeGemmaScaledWeights computes (1 + weight) for all RMSNorm layers
// This avoids creating temporary arrays on every forward pass
func precomputeGemmaScaledWeights(m *TextModel) {
	m.NormScaled = mlx.AddScalar(m.Norm.Weight, 1.0)

	for _, layer := range m.Layers {
		layer.InputNormScaled = mlx.AddScalar(layer.InputNorm.Weight, 1.0)
		layer.PostAttnNormScaled = mlx.AddScalar(layer.PostAttnNorm.Weight, 1.0)
		layer.PreFFNormScaled = mlx.AddScalar(layer.PreFFNorm.Weight, 1.0)
		layer.PostFFNormScaled = mlx.AddScalar(layer.PostFFNorm.Weight, 1.0)

		layer.Attention.QNormScaled = mlx.AddScalar(layer.Attention.QNorm.Weight, 1.0)
		layer.Attention.KNormScaled = mlx.AddScalar(layer.Attention.KNorm.Weight, 1.0)
	}

	// Eval all the precomputed weights
	var scaled []*mlx.Array
	scaled = append(scaled, m.NormScaled)
	for _, layer := range m.Layers {
		scaled = append(scaled, layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled,
			layer.Attention.QNormScaled, layer.Attention.KNormScaled)
	}
	mlx.Eval(scaled...)
}

// Forward runs the text model forward pass
func (m *TextModel) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]

	// Get embeddings and scale by sqrt(hidden_size)
	h := m.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(m.HiddenSize))))

	for i, layer := range m.Layers {
		h = layer.Forward(h, caches[i], B, L, m.TextConfig)
	}

	// Final norm and output projection
	return m.Output.Forward(mlx.RMSNorm(h, m.NormScaled, m.RMSNormEps))
}

// Forward runs a decoder layer
func (l *DecoderLayer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *TextConfig) *mlx.Array {
	// Pre-attention norm (use precomputed scaled weight)
	normed := mlx.RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)

	// Attention
	attnOut := l.Attention.Forward(normed, c, B, L, l.IsSliding, cfg)

	// Post-attention norm and residual
	attnOut = mlx.RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	// Pre-FFN norm
	normed = mlx.RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)

	// MLP
	mlpOut := l.MLP.Forward(normed)

	// Post-FFN norm and residual
	mlpOut = mlx.RMSNorm(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	return mlx.Add(h, mlpOut)
}

// Interface methods
func (m *TextModel) NumLayers() int          { return len(m.Layers) }
func (m *TextModel) MaxContextLength() int32 { return m.MaxPositionEmbeddings }
func (m *TextModel) VocabSize() int32        { return m.TextConfig.VocabSize }

// Tokenizer returns the tokenizer wrapped to add BOS and apply chat template
func (m *TextModel) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

// FormatPrompt applies the Gemma 3 chat template to a prompt
func (m *TextModel) FormatPrompt(prompt string) string {
	// Gemma 3 chat format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

func (m *TextModel) NewCache(maxSeqLen int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		if m.Layers[i].IsSliding {
			// Use rotating cache for sliding window layers
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			// Use regular cache for global attention layers
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}
