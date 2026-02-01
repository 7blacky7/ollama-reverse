//go:build mlx

// Modul: config.go
// Beschreibung: Konfigurationsstruktur für den Qwen3 Text-Encoder.
// Enthält: Config-Struct mit allen Modell-Parametern.

package qwen3

// Config holds Qwen3 text encoder configuration
type Config struct {
	HiddenSize        int32   `json:"hidden_size"`
	NumHiddenLayers   int32   `json:"num_hidden_layers"`
	IntermediateSize  int32   `json:"intermediate_size"`
	NumAttentionHeads int32   `json:"num_attention_heads"`
	NumKeyValueHeads  int32   `json:"num_key_value_heads"`
	VocabSize         int32   `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	HeadDim           int32   `json:"head_dim"`
}
