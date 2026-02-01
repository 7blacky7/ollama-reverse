// openai_types.go - Typdefinitionen fuer OpenAI-kompatible API
//
// Enthaelt:
// - Error und ErrorResponse Typen
// - Message, Choice und ChunkChoice Typen
// - Request- und Response-Strukturen (Chat, Completion, Embedding, Image)
// - Usage und weitere Hilfstypen
//
// Verwandte Dateien:
// - openai_to.go: Konvertierung API -> OpenAI Format
// - openai_from.go: Konvertierung OpenAI -> API Format
package openai

import (
	"encoding/json"
	"net/http"

	"github.com/ollama/ollama/api"
)

// finishReasonToolCalls wird verwendet wenn Tool-Calls vorhanden sind
var finishReasonToolCalls = "tool_calls"

// Error repraesentiert einen OpenAI-kompatiblen Fehler
type Error struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   any     `json:"param"`
	Code    *string `json:"code"`
}

// ErrorResponse ist die Wrapper-Struktur fuer Fehlerantworten
type ErrorResponse struct {
	Error Error `json:"error"`
}

// Message repraesentiert eine Chat-Nachricht
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Reasoning  string     `json:"reasoning,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ChoiceLogprobs enthaelt Log-Wahrscheinlichkeiten fuer Tokens
type ChoiceLogprobs struct {
	Content []api.Logprob `json:"content"`
}

// Choice repraesentiert eine Antwort-Option bei Chat-Completions
type Choice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

// ChunkChoice repraesentiert eine Antwort-Option beim Streaming
type ChunkChoice struct {
	Index        int             `json:"index"`
	Delta        Message         `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

// CompleteChunkChoice repraesentiert eine Antwort-Option bei Text-Completions
type CompleteChunkChoice struct {
	Text         string          `json:"text"`
	Index        int             `json:"index"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

// Usage enthaelt Token-Verbrauchsinformationen
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ResponseFormat definiert das gewuenschte Antwortformat
type ResponseFormat struct {
	Type       string      `json:"type"`
	JsonSchema *JsonSchema `json:"json_schema,omitempty"`
}

// JsonSchema fuer strukturierte JSON-Antworten
type JsonSchema struct {
	Schema json.RawMessage `json:"schema"`
}

// EmbedRequest ist ein Request fuer Embeddings
type EmbedRequest struct {
	Input          any    `json:"input"`
	Model          string `json:"model"`
	Dimensions     int    `json:"dimensions,omitempty"`
	EncodingFormat string `json:"encoding_format,omitempty"` // "float" or "base64"
}

// StreamOptions fuer Streaming-Konfiguration
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Reasoning fuer Reasoning-Konfiguration
type Reasoning struct {
	Effort string `json:"effort,omitempty"`
}

// ChatCompletionRequest ist ein Request fuer Chat-Completions
type ChatCompletionRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Stream           bool            `json:"stream"`
	StreamOptions    *StreamOptions  `json:"stream_options"`
	MaxTokens        *int            `json:"max_tokens"`
	Seed             *int            `json:"seed"`
	Stop             any             `json:"stop"`
	Temperature      *float64        `json:"temperature"`
	FrequencyPenalty *float64        `json:"frequency_penalty"`
	PresencePenalty  *float64        `json:"presence_penalty"`
	TopP             *float64        `json:"top_p"`
	ResponseFormat   *ResponseFormat `json:"response_format"`
	Tools            []api.Tool      `json:"tools"`
	Reasoning        *Reasoning      `json:"reasoning,omitempty"`
	ReasoningEffort  *string         `json:"reasoning_effort,omitempty"`
	Logprobs         *bool           `json:"logprobs"`
	TopLogprobs      int             `json:"top_logprobs"`
	DebugRenderOnly  bool            `json:"_debug_render_only"`
}

// ChatCompletion ist die Antwort fuer Chat-Completions
type ChatCompletion struct {
	Id                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint"`
	Choices           []Choice       `json:"choices"`
	Usage             Usage          `json:"usage,omitempty"`
	DebugInfo         *api.DebugInfo `json:"_debug_info,omitempty"`
}

// ChatCompletionChunk ist ein Streaming-Chunk fuer Chat-Completions
type ChatCompletionChunk struct {
	Id                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	SystemFingerprint string        `json:"system_fingerprint"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
}

// CompletionRequest ist ein Request fuer Text-Completions
// TODO (https://github.com/ollama/ollama/issues/5259): support []string, []int and [][]int
type CompletionRequest struct {
	Model            string         `json:"model"`
	Prompt           string         `json:"prompt"`
	FrequencyPenalty float32        `json:"frequency_penalty"`
	MaxTokens        *int           `json:"max_tokens"`
	PresencePenalty  float32        `json:"presence_penalty"`
	Seed             *int           `json:"seed"`
	Stop             any            `json:"stop"`
	Stream           bool           `json:"stream"`
	StreamOptions    *StreamOptions `json:"stream_options"`
	Temperature      *float32       `json:"temperature"`
	TopP             float32        `json:"top_p"`
	Suffix           string         `json:"suffix"`
	Logprobs         *int           `json:"logprobs"`
	DebugRenderOnly  bool           `json:"_debug_render_only"`
}

// Completion ist die Antwort fuer Text-Completions
type Completion struct {
	Id                string                `json:"id"`
	Object            string                `json:"object"`
	Created           int64                 `json:"created"`
	Model             string                `json:"model"`
	SystemFingerprint string                `json:"system_fingerprint"`
	Choices           []CompleteChunkChoice `json:"choices"`
	Usage             Usage                 `json:"usage,omitempty"`
}

// CompletionChunk ist ein Streaming-Chunk fuer Text-Completions
type CompletionChunk struct {
	Id                string                `json:"id"`
	Object            string                `json:"object"`
	Created           int64                 `json:"created"`
	Choices           []CompleteChunkChoice `json:"choices"`
	Model             string                `json:"model"`
	SystemFingerprint string                `json:"system_fingerprint"`
	Usage             *Usage                `json:"usage,omitempty"`
}

// ToolCall repraesentiert einen Tool-Aufruf
type ToolCall struct {
	ID       string `json:"id"`
	Index    int    `json:"index"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Model repraesentiert ein verfuegbares Modell
type Model struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// Embedding repraesentiert ein einzelnes Embedding-Ergebnis
type Embedding struct {
	Object    string `json:"object"`
	Embedding any    `json:"embedding"` // Can be []float32 (float format) or string (base64 format)
	Index     int    `json:"index"`
}

// ListCompletion ist die Antwort fuer Model-Listen
type ListCompletion struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// EmbeddingList ist die Antwort fuer Embedding-Requests
type EmbeddingList struct {
	Object string         `json:"object"`
	Data   []Embedding    `json:"data"`
	Model  string         `json:"model"`
	Usage  EmbeddingUsage `json:"usage,omitempty"`
}

// EmbeddingUsage enthaelt Token-Verbrauch fuer Embeddings
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ImageGenerationRequest ist ein OpenAI-kompatibler Image-Generation Request
type ImageGenerationRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	Seed           *int64 `json:"seed,omitempty"`
}

// ImageGenerationResponse ist die Antwort fuer Image-Generation
type ImageGenerationResponse struct {
	Created int64            `json:"created"`
	Data    []ImageURLOrData `json:"data"`
}

// ImageURLOrData enthaelt entweder eine URL oder base64-kodierte Bilddaten
type ImageURLOrData struct {
	URL     string `json:"url,omitempty"`
	B64JSON string `json:"b64_json,omitempty"`
}

// ImageEditRequest ist ein OpenAI-kompatibler Image-Edit Request
type ImageEditRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Image  string `json:"image"`          // Base64-encoded image data
	Size   string `json:"size,omitempty"` // e.g., "1024x1024"
	Seed   *int64 `json:"seed,omitempty"`
}

// NewError erstellt eine neue ErrorResponse basierend auf HTTP-Statuscode
func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{Error{Type: etype, Message: message}}
}
