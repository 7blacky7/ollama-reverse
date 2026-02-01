//go:build mlx

// config.go - Konfigurationsstrukturen fuer Gemma 3.
//
// Dieses Modul enthaelt:
// - TextConfig fuer das Text-Modell
// - Config fuer das multimodale Modell
// - Strukturdefinitionen fuer Decoder-Layer, Attention und MLP

package gemma3

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// TextConfig holds configuration for the text model
type TextConfig struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeLocalBaseFreq     float32 `json:"rope_local_base_freq"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	SlidingWindow         int32   `json:"sliding_window"`
	SlidingWindowPattern  int32   `json:"sliding_window_pattern"`

	// Computed fields
	Scale float32 `json:"-"`
}

// Config holds config for the full multimodal model
type Config struct {
	TextConfig   TextConfig   `json:"text_config"`
	VisionConfig VisionConfig `json:"vision_config"`

	// Image token config (from config.json)
	BOITokenIndex    int32 `json:"boi_token_index"`    // <start_of_image> = 255999
	EOITokenIndex    int32 `json:"eoi_token_index"`    // <end_of_image> = 256000
	ImageTokenIndex  int32 `json:"image_token_index"`  // <image_soft_token> = 262144
	MMTokensPerImage int32 `json:"mm_tokens_per_image"` // 256
}

// DecoderLayer is a single transformer block
type DecoderLayer struct {
	InputNorm    *nn.RMSNorm `weight:"input_layernorm"`
	Attention    *Attention
	PostAttnNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
	PreFFNorm    *nn.RMSNorm `weight:"pre_feedforward_layernorm"`
	MLP          *MLP
	PostFFNorm   *nn.RMSNorm `weight:"post_feedforward_layernorm"`

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	InputNormScaled    *mlx.Array `weight:"-"`
	PostAttnNormScaled *mlx.Array `weight:"-"`
	PreFFNormScaled    *mlx.Array `weight:"-"`
	PostFFNormScaled   *mlx.Array `weight:"-"`

	// Whether this layer uses sliding window attention
	IsSliding bool
	LayerIdx  int32
}

// Attention implements Gemma 3 attention with Q/K normalization
type Attention struct {
	QProj *nn.Linear  `weight:"self_attn.q_proj"`
	KProj *nn.Linear  `weight:"self_attn.k_proj"`
	VProj *nn.Linear  `weight:"self_attn.v_proj"`
	OProj *nn.Linear  `weight:"self_attn.o_proj"`
	QNorm *nn.RMSNorm `weight:"self_attn.q_norm"`
	KNorm *nn.RMSNorm `weight:"self_attn.k_norm"`

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	QNormScaled *mlx.Array `weight:"-"`
	KNormScaled *mlx.Array `weight:"-"`
}

// MLP is the feed-forward network with GELU activation
type MLP struct {
	GateProj *nn.Linear `weight:"mlp.gate_proj"`
	UpProj   *nn.Linear `weight:"mlp.up_proj"`
	DownProj *nn.Linear `weight:"mlp.down_proj"`
}

// MultiModalProjector projects vision features to text space
type MultiModalProjector struct {
	SoftEmbNorm   *nn.RMSNorm `weight:"soft_emb_norm"`
	Avg           *nn.Linear  `weight:"avg_pool.proj"` // Pool 4 patches to 1
	Linear        *nn.Linear  `weight:"mm_soft_emb_proj"`

	// Precomputed scaled weight for Gemma-style RMSNorm
	SoftEmbNormScaled *mlx.Array `weight:"-"`
}

// isLayerSliding determines if a layer uses sliding window attention
// Pattern N means: layers 0 to N-1 sliding, N full, N+1 to 2N-1 sliding, 2N full, etc.
func isLayerSliding(layerIdx, pattern int32) bool {
	if pattern <= 0 {
		return false // No sliding window
	}
	// Layer is full attention if (layerIdx + 1) % pattern == 0
	return (layerIdx+1)%pattern != 0
}
