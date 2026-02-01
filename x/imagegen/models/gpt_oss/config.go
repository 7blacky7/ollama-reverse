//go:build mlx

// config.go - Konfigurationsstrukturen fuer GPT-OSS
// Enthaelt Config und RopeScaling fuer das Modell.
package gpt_oss

// RopeScaling holds YaRN or other RoPE scaling configuration
type RopeScaling struct {
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
}

// Config holds the model configuration
type Config struct {
	HiddenSize        int32        `json:"hidden_size"`
	NumHiddenLayers   int32        `json:"num_hidden_layers"`
	IntermediateSize  int32        `json:"intermediate_size"`
	NumAttentionHeads int32        `json:"num_attention_heads"`
	NumKeyValueHeads  int32        `json:"num_key_value_heads"`
	VocabSize         int32        `json:"vocab_size"`
	RMSNormEps        float32      `json:"rms_norm_eps"`
	RopeTheta         float32      `json:"rope_theta"`
	HeadDim           int32        `json:"head_dim"`
	SlidingWindow     int32        `json:"sliding_window"`
	NumLocalExperts   int32        `json:"num_local_experts"`
	NumExpertsPerTok  int32        `json:"num_experts_per_tok"`
	LayerTypes        []string     `json:"layer_types"`
	SwiGLULimit       float32      `json:"swiglu_limit"`
	RopeScaling       *RopeScaling `json:"rope_scaling"`
	Scale             float32      `json:"-"` // computed: 1/sqrt(HeadDim)
}
