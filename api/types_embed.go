// types_embed.go - Embedding API Types
// Enthaelt: EmbedRequest, EmbedResponse, EmbeddingRequest, EmbeddingResponse
package api

import "time"

// EmbedRequest is the request passed to [Client.Embed].
type EmbedRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Input is the input to embed.
	Input any `json:"input"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Truncate truncates the input to fit the model's max sequence length.
	Truncate *bool `json:"truncate,omitempty"`

	// Dimensions truncates the output embedding to the specified dimension.
	Dimensions int `json:"dimensions,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbedResponse is the response from [Client.Embed].
type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`

	TotalDuration   time.Duration `json:"total_duration,omitempty"`
	LoadDuration    time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
}

// EmbeddingRequest is the request passed to [Client.Embeddings].
// Deprecated: Use EmbedRequest instead.
type EmbeddingRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Prompt is the textual prompt to embed.
	Prompt string `json:"prompt"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbeddingResponse is the response from [Client.Embeddings].
// Deprecated: Use EmbedResponse instead.
type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}
