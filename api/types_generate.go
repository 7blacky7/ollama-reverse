// types_generate.go - Generate und Chat API Types
// Enthaelt: GenerateRequest, GenerateResponse, ChatRequest, ChatResponse, Message, ThinkValue, Logprob
package api

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// GenerateRequest describes a request sent by [Client.Generate]. While you
// have to specify the Model and Prompt fields, all the other fields have
// reasonable defaults for basic uses.
type GenerateRequest struct {
	// Model is the model name; it should be a name familiar to Ollama from
	// the library at https://ollama.com/library
	Model string `json:"model"`

	// Prompt is the textual prompt to send to the model.
	Prompt string `json:"prompt"`

	// Suffix is the text that comes after the inserted text.
	Suffix string `json:"suffix"`

	// System overrides the model's default system message/prompt.
	System string `json:"system"`

	// Template overrides the model's default prompt template.
	Template string `json:"template"`

	// Context is the context parameter returned from a previous call to
	// [Client.Generate]. It can be used to keep a short conversational memory.
	Context []int `json:"context,omitempty"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Raw set to true means that no formatting will be applied to the prompt.
	Raw bool `json:"raw,omitempty"`

	// Format specifies the format to return a response in.
	Format json.RawMessage `json:"format,omitempty"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Images is an optional list of raw image bytes accompanying this
	// request, for multimodal models.
	Images []ImageData `json:"images,omitempty"`

	// Options lists model-specific options. For example, temperature can be
	// set through this field, if the model supports it.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding.
	Think *ThinkValue `json:"think,omitempty"`

	// Truncate truncates the chat history if the prompt exceeds context length.
	Truncate *bool `json:"truncate,omitempty"`

	// Shift shifts the chat history when hitting the context length limit.
	Shift *bool `json:"shift,omitempty"`

	// DebugRenderOnly returns the rendered template instead of calling the model.
	DebugRenderOnly bool `json:"_debug_render_only,omitempty"`

	// Logprobs specifies whether to return log probabilities of the output tokens.
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number of most likely tokens to return at each position.
	TopLogprobs int `json:"top_logprobs,omitempty"`

	// Experimental: Image generation fields
	Width  int32 `json:"width,omitempty"`
	Height int32 `json:"height,omitempty"`
	Steps  int32 `json:"steps,omitempty"`
}

// GenerateResponse is the response passed into [GenerateResponseFunc].
type GenerateResponse struct {
	Model       string    `json:"model"`
	RemoteModel string    `json:"remote_model,omitempty"`
	RemoteHost  string    `json:"remote_host,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	Response    string    `json:"response"`
	Thinking    string    `json:"thinking,omitempty"`
	Done        bool      `json:"done"`
	DoneReason  string    `json:"done_reason,omitempty"`
	Context     []int     `json:"context,omitempty"`

	Metrics

	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	DebugInfo *DebugInfo `json:"_debug_info,omitempty"`
	Logprobs  []Logprob  `json:"logprobs,omitempty"`

	// Experimental: Image generation fields
	Image     string `json:"image,omitempty"`
	Completed int64  `json:"completed,omitempty"`
	Total     int64  `json:"total,omitempty"`
}

// ChatRequest describes a request sent by [Client.Chat].
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive *Duration       `json:"keep_alive,omitempty"`
	Tools     `json:"tools,omitempty"`
	Options   map[string]any `json:"options"`

	Think           *ThinkValue `json:"think,omitempty"`
	Truncate        *bool       `json:"truncate,omitempty"`
	Shift           *bool       `json:"shift,omitempty"`
	DebugRenderOnly bool        `json:"_debug_render_only,omitempty"`
	Logprobs        bool        `json:"logprobs,omitempty"`
	TopLogprobs     int         `json:"top_logprobs,omitempty"`
}

// ChatResponse is the response returned by [Client.Chat].
type ChatResponse struct {
	Model       string    `json:"model"`
	RemoteModel string    `json:"remote_model,omitempty"`
	RemoteHost  string    `json:"remote_host,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	Message     Message   `json:"message"`
	Done        bool      `json:"done"`
	DoneReason  string    `json:"done_reason,omitempty"`
	DebugInfo   *DebugInfo `json:"_debug_info,omitempty"`
	Logprobs    []Logprob  `json:"logprobs,omitempty"`

	Metrics
}

// DebugInfo contains debug information for template rendering
type DebugInfo struct {
	RenderedTemplate string `json:"rendered_template"`
	ImageCount       int    `json:"image_count,omitempty"`
}

// Message is a single message in a chat sequence.
type Message struct {
	Role       string      `json:"role"`
	Content    string      `json:"content"`
	Thinking   string      `json:"thinking,omitempty"`
	Images     []ImageData `json:"images,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolName   string      `json:"tool_name,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	type Alias Message
	var a Alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}

	*m = Message(a)
	m.Role = strings.ToLower(m.Role)
	return nil
}

// TokenLogprob represents log probability information for a single token.
type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Logprob contains log probability information for a generated token.
type Logprob struct {
	TokenLogprob
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

// ThinkValue represents a value that can be a boolean or a string ("high", "medium", "low")
type ThinkValue struct {
	Value interface{}
}

// IsValid checks if the ThinkValue is valid
func (t *ThinkValue) IsValid() bool {
	if t == nil || t.Value == nil {
		return true
	}

	switch v := t.Value.(type) {
	case bool:
		return true
	case string:
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// IsBool returns true if the value is a boolean
func (t *ThinkValue) IsBool() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(bool)
	return ok
}

// IsString returns true if the value is a string
func (t *ThinkValue) IsString() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(string)
	return ok
}

// Bool returns the value as a bool (true if enabled in any way)
func (t *ThinkValue) Bool() bool {
	if t == nil || t.Value == nil {
		return false
	}

	switch v := t.Value.(type) {
	case bool:
		return v
	case string:
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// String returns the value as a string
func (t *ThinkValue) String() string {
	if t == nil || t.Value == nil {
		return ""
	}

	switch v := t.Value.(type) {
	case string:
		return v
	case bool:
		if v {
			return "medium"
		}
		return ""
	default:
		return ""
	}
}

// UnmarshalJSON implements json.Unmarshaler
func (t *ThinkValue) UnmarshalJSON(data []byte) error {
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.Value = b
		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		if s != "high" && s != "medium" && s != "low" {
			return fmt.Errorf("invalid think value: %q (must be \"high\", \"medium\", \"low\", true, or false)", s)
		}
		t.Value = s
		return nil
	}

	return fmt.Errorf("think must be a boolean or string (\"high\", \"medium\", \"low\", true, or false)")
}

// MarshalJSON implements json.Marshaler
func (t *ThinkValue) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}
