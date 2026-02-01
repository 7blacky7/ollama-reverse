//go:build mlx

// Package flux2 - Haupt-Transformer-Modell
// Enthält: Flux2Transformer2DModel, Load, Forward

package flux2

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// Flux2Transformer2DModel is the main Flux2 transformer
// Weight names at top level: time_guidance_embed.*, double_stream_modulation_*.*, etc.
type Flux2Transformer2DModel struct {
	// Timestep embedding
	TimeGuidanceEmbed *TimeGuidanceEmbed `weight:"time_guidance_embed"`

	// Shared modulation
	DoubleStreamModulationImg *Modulation `weight:"double_stream_modulation_img"`
	DoubleStreamModulationTxt *Modulation `weight:"double_stream_modulation_txt"`
	SingleStreamModulation    *Modulation `weight:"single_stream_modulation"`

	// Embedders
	XEmbedder       nn.LinearLayer `weight:"x_embedder"`
	ContextEmbedder nn.LinearLayer `weight:"context_embedder"`

	// Transformer blocks
	TransformerBlocks       []*TransformerBlock       `weight:"transformer_blocks"`
	SingleTransformerBlocks []*SingleTransformerBlock `weight:"single_transformer_blocks"`

	// Output
	NormOut *NormOut       `weight:"norm_out"`
	ProjOut nn.LinearLayer `weight:"proj_out"`

	*TransformerConfig
}

// Load loads the Flux2 transformer from ollama blob storage.
func (m *Flux2Transformer2DModel) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading transformer... ")

	// Load config from blob
	var cfg TransformerConfig
	if err := manifest.ReadConfigJSON("transformer/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.TransformerConfig = &cfg

	// Initialize slices
	m.TransformerBlocks = make([]*TransformerBlock, cfg.NumLayers)
	m.SingleTransformerBlocks = make([]*SingleTransformerBlock, cfg.NumSingleLayers)

	// Initialize TimeGuidanceEmbed with embed dim
	m.TimeGuidanceEmbed = &TimeGuidanceEmbed{
		TimestepEmbedder: &TimestepEmbedder{EmbedDim: cfg.TimestepGuidanceChannels},
	}

	// Load weights from tensor blobs
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
func (m *Flux2Transformer2DModel) loadWeights(weights safetensors.WeightSource) error {
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	m.initComputedFields()
	fmt.Println("\u2713")
	return nil
}

// initComputedFields initializes computed fields after loading weights
func (m *Flux2Transformer2DModel) initComputedFields() {
	cfg := m.TransformerConfig
	innerDim := cfg.InnerDim()
	scale := float32(1.0 / math.Sqrt(float64(cfg.AttentionHeadDim)))

	// Initialize transformer blocks
	for _, block := range m.TransformerBlocks {
		block.NHeads = cfg.NumAttentionHeads
		block.HeadDim = cfg.AttentionHeadDim
		block.Scale = scale
	}

	// Initialize single transformer blocks
	for _, block := range m.SingleTransformerBlocks {
		block.NHeads = cfg.NumAttentionHeads
		block.HeadDim = cfg.AttentionHeadDim
		block.InnerDim = innerDim
		block.MLPHidDim = cfg.MLPHiddenDim()
		block.Scale = scale
	}
}

// Forward runs the Flux2 transformer
func (m *Flux2Transformer2DModel) Forward(patches, txtEmbeds *mlx.Array, timesteps *mlx.Array, rope *RoPECache) *mlx.Array {
	patchShape := patches.Shape()
	B := patchShape[0]
	imgLen := patchShape[1]
	txtLen := txtEmbeds.Shape()[1]

	// Scale timestep to 0-1000 range (diffusers multiplies by 1000)
	scaledTimesteps := mlx.MulScalar(timesteps, 1000.0)

	// Compute timestep embedding
	temb := m.TimeGuidanceEmbed.Forward(scaledTimesteps)

	// Embed patches and text
	imgHidden := m.XEmbedder.Forward(patches)
	txtHidden := m.ContextEmbedder.Forward(txtEmbeds)

	// Compute shared modulation
	imgMod := m.DoubleStreamModulationImg.Forward(temb)
	txtMod := m.DoubleStreamModulationTxt.Forward(temb)
	singleMod := m.SingleStreamModulation.Forward(temb)

	// Double (dual-stream) blocks
	for _, block := range m.TransformerBlocks {
		imgHidden, txtHidden = block.Forward(imgHidden, txtHidden, imgMod, txtMod, rope.Cos, rope.Sin)
	}

	// Concatenate for single-stream: text first, then image
	hidden := mlx.Concatenate([]*mlx.Array{txtHidden, imgHidden}, 1)

	// Single-stream blocks
	for _, block := range m.SingleTransformerBlocks {
		hidden = block.Forward(hidden, singleMod, rope.Cos, rope.Sin)
	}

	// Extract image portion
	totalLen := txtLen + imgLen
	imgOut := mlx.Slice(hidden, []int32{0, txtLen, 0}, []int32{B, totalLen, hidden.Shape()[2]})

	// Final norm and projection
	imgOut = m.NormOut.Forward(imgOut, temb)
	return m.ProjOut.Forward(imgOut)
}
