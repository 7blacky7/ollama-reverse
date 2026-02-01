// responses_types.go - OpenAI Responses API Type-Definitionen
//
// Dieses Modul enthält alle Type-Definitionen für die Responses API:
// - ResponsesContent (Text, Image, OutputText)
// - ResponsesInputMessage, ResponsesInputItem
// - ResponsesFunctionCall, ResponsesFunctionCallOutput
// - ResponsesReasoningInput
// - ResponsesInput, ResponsesReasoning, ResponsesTool
// - ResponsesRequest
//
// Siehe responses_unmarshal.go für JSON Unmarshal-Funktionen

package openai

import "encoding/json"

// ResponsesContent is a discriminated union for input content types.
type ResponsesContent interface {
	responsesContent()
}

type ResponsesTextContent struct {
	Type string `json:"type"` // always "input_text"
	Text string `json:"text"`
}

func (ResponsesTextContent) responsesContent() {}

type ResponsesImageContent struct {
	Type     string `json:"type"`              // always "input_image"
	Detail   string `json:"detail"`            // required
	FileID   string `json:"file_id,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

func (ResponsesImageContent) responsesContent() {}

// ResponsesOutputTextContent represents output text from a previous assistant response.
type ResponsesOutputTextContent struct {
	Type string `json:"type"` // always "output_text"
	Text string `json:"text"`
}

func (ResponsesOutputTextContent) responsesContent() {}

type ResponsesInputMessage struct {
	Type    string             `json:"type"` // always "message"
	Role    string             `json:"role"` // one of `user`, `system`, `developer`
	Content []ResponsesContent `json:"content,omitempty"`
}

type ResponsesOutputMessage struct{}

// ResponsesInputItem is a discriminated union for input items.
type ResponsesInputItem interface {
	responsesInputItem()
}

func (ResponsesInputMessage) responsesInputItem() {}

// ResponsesFunctionCall represents an assistant's function call in conversation history.
type ResponsesFunctionCall struct {
	ID        string `json:"id,omitempty"`
	Type      string `json:"type"`      // always "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (ResponsesFunctionCall) responsesInputItem() {}

// ResponsesFunctionCallOutput represents a function call result from the client.
type ResponsesFunctionCallOutput struct {
	Type   string `json:"type"`    // always "function_call_output"
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

func (ResponsesFunctionCallOutput) responsesInputItem() {}

// ResponsesReasoningInput represents a reasoning item passed back as input.
type ResponsesReasoningInput struct {
	ID               string                      `json:"id,omitempty"`
	Type             string                      `json:"type"` // always "reasoning"
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent string                      `json:"encrypted_content,omitempty"`
}

func (ResponsesReasoningInput) responsesInputItem() {}

// ResponsesInput can be either a string or an array of input items.
type ResponsesInput struct {
	Text  string
	Items []ResponsesInputItem
}

type ResponsesReasoning struct {
	Effort          string `json:"effort,omitempty"`
	GenerateSummary string `json:"generate_summary,omitempty"` // deprecated
	Summary         string `json:"summary,omitempty"`
}

type ResponsesTextFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict *bool           `json:"strict,omitempty"`
}

type ResponsesText struct {
	Format *ResponsesTextFormat `json:"format,omitempty"`
}

// ResponsesTool represents a tool in the Responses API format.
type ResponsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description *string        `json:"description"`
	Strict      *bool          `json:"strict"`
	Parameters  map[string]any `json:"parameters"`
}

type ResponsesRequest struct {
	Model        string          `json:"model"`
	Background   bool            `json:"background"`
	Conversation json.RawMessage `json:"conversation"`
	Include      []string        `json:"include"`
	Input        ResponsesInput  `json:"input"`
	Instructions string          `json:"instructions,omitempty"`

	MaxOutputTokens *int               `json:"max_output_tokens,omitempty"`
	Reasoning       ResponsesReasoning `json:"reasoning"`
	Temperature     *float64           `json:"temperature"`
	Text            *ResponsesText     `json:"text,omitempty"`
	TopP            *float64           `json:"top_p"`
	Truncation      *string            `json:"truncation"`
	Tools           []ResponsesTool    `json:"tools,omitempty"`
	Stream          *bool              `json:"stream,omitempty"`
}
