//go:build mlx

// multimodal.go - Multimodales Gemma 3 Modell.
//
// Dieses Modul enthaelt:
// - Model Struktur fuer das vollstaendige multimodale Modell
// - Load-Funktion mit Konfigurationshandling
// - Forward-Pass mit Bildverarbeitung
// - Image-Token-Expansion und -Einbettung

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

// Model is the full Gemma 3 multimodal model
type Model struct {
	VisionTower *VisionTower         `weight:"vision_tower"`
	Projector   *MultiModalProjector `weight:"multi_modal_projector"`
	TextModel   *TextModel           `weight:"language_model"`
	Config      *Config
	tok         *tokenizer.Tokenizer
}

// Load loads the full multimodal Gemma 3 model
func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Set defaults for text config (multimodal config often has incomplete text_config)
	// These defaults match transformers.Gemma3TextConfig defaults
	tc := &cfg.TextConfig
	if tc.HeadDim == 0 {
		tc.HeadDim = 256 // Gemma 3 uses head_dim=256
	}
	if tc.NumAttentionHeads == 0 {
		// Gemma 3 4B uses 8 attention heads (cannot infer from hidden_size/head_dim)
		tc.NumAttentionHeads = 8
	}
	if tc.NumKeyValueHeads == 0 {
		// Gemma 3 4B uses 4 KV heads (GQA with 2:1 ratio)
		tc.NumKeyValueHeads = 4
	}
	if tc.VocabSize == 0 {
		tc.VocabSize = 262208 // Gemma 3 vocab size (not 262144!)
	}
	if tc.RopeTheta == 0 {
		tc.RopeTheta = 1000000
	}
	if tc.RopeLocalBaseFreq == 0 {
		tc.RopeLocalBaseFreq = 10000
	}
	if tc.RMSNormEps == 0 {
		tc.RMSNormEps = 1e-6
	}
	if tc.SlidingWindowPattern == 0 {
		tc.SlidingWindowPattern = 6
	}
	if tc.MaxPositionEmbeddings == 0 {
		tc.MaxPositionEmbeddings = 131072 // Gemma 3 4B default
	}

	// Compute text model scale
	tc.Scale = float32(1.0 / math.Sqrt(float64(tc.HeadDim)))

	// Set defaults for image token config
	if cfg.BOITokenIndex == 0 {
		cfg.BOITokenIndex = 255999 // <start_of_image>
	}
	if cfg.EOITokenIndex == 0 {
		cfg.EOITokenIndex = 256000 // <end_of_image>
	}
	if cfg.ImageTokenIndex == 0 {
		cfg.ImageTokenIndex = 262144 // <image_soft_token>
	}
	if cfg.MMTokensPerImage == 0 {
		cfg.MMTokensPerImage = 256
	}

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	m := &Model{
		VisionTower: &VisionTower{
			Embeddings: &VisionEmbeddings{},
			Encoder:    make([]*VisionEncoderLayer, cfg.VisionConfig.NumHiddenLayers),
			Config:     &cfg.VisionConfig,
		},
		Projector: &MultiModalProjector{},
		TextModel: &TextModel{
			Layers:     make([]*DecoderLayer, cfg.TextConfig.NumHiddenLayers),
			TextConfig: &cfg.TextConfig,
		},
		Config: &cfg,
		tok:    tok,
	}

	// Initialize text layer metadata
	for i := range m.TextModel.Layers {
		m.TextModel.Layers[i] = &DecoderLayer{
			LayerIdx:  int32(i),
			IsSliding: isLayerSliding(int32(i), cfg.TextConfig.SlidingWindowPattern),
		}
	}

	// Initialize vision encoder layers
	for i := range m.VisionTower.Encoder {
		m.VisionTower.Encoder[i] = &VisionEncoderLayer{}
	}

	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Tied embeddings for text output
	m.TextModel.Output = nn.NewLinear(m.TextModel.EmbedTokens.Weight, nil)
	m.TextModel.tok = tok

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	// Precompute (1 + weight) for Gemma-style RMSNorm
	precomputeGemmaScaledWeights(m.TextModel)

	// Precompute projector's scaled weight
	m.Projector.SoftEmbNormScaled = mlx.AddScalar(m.Projector.SoftEmbNorm.Weight, 1.0)
	mlx.Eval(m.Projector.SoftEmbNormScaled)

	return m, nil
}

// Forward runs the text-only forward pass
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	return m.TextModel.Forward(tokens, caches)
}

// ForwardWithImage runs the multimodal forward pass
// tokens: [B, L] input token IDs (with image placeholder tokens)
// image: [B, H, W, C] preprocessed image tensor
func (m *Model) ForwardWithImage(tokens *mlx.Array, image *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	cfg := m.Config.TextConfig

	// Find image token position FIRST before any eval that might free tokens
	imageStartPos := int32(-1)
	if image != nil && B == 1 {
		tokenData := tokens.DataInt32() // This evals tokens
		for i, t := range tokenData {
			if t == m.Config.ImageTokenIndex {
				imageStartPos = int32(i)
				break
			}
		}
	}

	// Get text embeddings and scale
	h := m.TextModel.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(cfg.HiddenSize))))

	// Process image if provided
	if image != nil && imageStartPos >= 0 {
		// Vision tower: [B, H, W, C] -> [B, num_patches, vision_hidden]
		visionFeatures := m.VisionTower.Forward(image)

		// Project to text space: [B, num_patches, vision_hidden] -> [B, 256, text_hidden]
		imageEmbeds := m.Projector.Forward(visionFeatures, cfg.RMSNormEps)

		// Eval h and imageEmbeds together so neither gets freed
		mlx.Eval(h, imageEmbeds)

		// Cast imageEmbeds to match text embeddings dtype (bf16)
		if imageEmbeds.Dtype() != h.Dtype() {
			imageEmbeds = mlx.AsType(imageEmbeds, h.Dtype())
			mlx.Eval(imageEmbeds)
		}

		// Insert image embeddings at the known position
		h = m.insertImageEmbeddingsAt(h, imageEmbeds, imageStartPos)
	}

	// Run through text model layers
	for i, layer := range m.TextModel.Layers {
		h = layer.Forward(h, caches[i], B, L, m.TextModel.TextConfig)
	}

	// Final norm and output projection
	return m.TextModel.Output.Forward(mlx.RMSNorm(h, m.TextModel.NormScaled, cfg.RMSNormEps))
}

// insertImageEmbeddingsAt replaces image placeholder tokens with actual image embeddings
// at a known position (to avoid re-scanning tokens after eval)
// textEmbeds: [B, L, hidden_size] text embeddings
// imageEmbeds: [B, 256, hidden_size] image embeddings from projector
// startPos: starting position of image tokens in the sequence
func (m *Model) insertImageEmbeddingsAt(textEmbeds, imageEmbeds *mlx.Array, startPos int32) *mlx.Array {
	numImageTokens := imageEmbeds.Shape()[1]
	L := textEmbeds.Shape()[1]

	// Split text embeddings: [0:startPos] + imageEmbeds + [startPos+256:L]
	afterStart := startPos + numImageTokens

	// Slice before image tokens: textEmbeds[:, 0:startPos, :]
	before := mlx.SliceAxis(textEmbeds, 1, 0, startPos)

	// Slice after image tokens: textEmbeds[:, startPos+256:L, :]
	after := mlx.SliceAxis(textEmbeds, 1, afterStart, L)

	// Concatenate: before + imageEmbeds + after along axis 1
	return mlx.Concatenate([]*mlx.Array{before, imageEmbeds, after}, 1)
}

// Interface methods for Model
func (m *Model) NumLayers() int                         { return len(m.TextModel.Layers) }
func (m *Model) MaxContextLength() int32                { return m.Config.TextConfig.MaxPositionEmbeddings }
func (m *Model) VocabSize() int32                       { return m.Config.TextConfig.VocabSize }
func (m *Model) Tokenizer() *tokenizer.Tokenizer        { return m.tok }
func (m *Model) NewCache(maxSeqLen int32) []cache.Cache { return m.TextModel.NewCache(maxSeqLen) }
func (m *Model) ImageSize() int32                       { return m.Config.VisionConfig.ImageSize }

// FormatPrompt applies the Gemma 3 multimodal chat template
func (m *Model) FormatPrompt(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

// FormatPromptWithImage applies the Gemma 3 multimodal chat template with image
func (m *Model) FormatPromptWithImage(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n<start_of_image>%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

// ExpandImageTokens expands <start_of_image> into 256 image placeholder tokens
// Input tokens containing boi_token (255999) are expanded to:
// boi_token + 256 * image_token + eoi_token
func (m *Model) ExpandImageTokens(tokens []int32) []int32 {
	result := make([]int32, 0, len(tokens)+int(m.Config.MMTokensPerImage)+1)

	for _, t := range tokens {
		if t == m.Config.BOITokenIndex {
			// Expand: boi + 256 * image_token + eoi
			result = append(result, m.Config.BOITokenIndex)
			for i := int32(0); i < m.Config.MMTokensPerImage; i++ {
				result = append(result, m.Config.ImageTokenIndex)
			}
			result = append(result, m.Config.EOITokenIndex)
		} else {
			result = append(result, t)
		}
	}

	return result
}
