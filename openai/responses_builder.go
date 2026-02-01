// responses_builder.go - Response Types und ToResponse Konvertierung
//
// Dieses Modul enthält:
// - Response Types (ResponsesResponse, ResponsesOutputItem, etc.)
// - ToResponse Konvertierung für nicht-streaming Antworten
// - Hilfsfunktionen (derefFloat64)

package openai

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

// ResponsesTextField represents the text output configuration in the response.
type ResponsesTextField struct {
	Format ResponsesTextFormat `json:"format"`
}

// ResponsesReasoningOutput represents reasoning configuration in the response.
type ResponsesReasoningOutput struct {
	Effort  *string `json:"effort,omitempty"`
	Summary *string `json:"summary,omitempty"`
}

// ResponsesError represents an error in the response.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesIncompleteDetails represents details about why a response was incomplete.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

type ResponsesResponse struct {
	ID                 string                      `json:"id"`
	Object             string                      `json:"object"`
	CreatedAt          int64                       `json:"created_at"`
	CompletedAt        *int64                      `json:"completed_at"`
	Status             string                      `json:"status"`
	IncompleteDetails  *ResponsesIncompleteDetails `json:"incomplete_details"`
	Model              string                      `json:"model"`
	PreviousResponseID *string                     `json:"previous_response_id"`
	Instructions       *string                     `json:"instructions"`
	Output             []ResponsesOutputItem       `json:"output"`
	Error              *ResponsesError             `json:"error"`
	Tools              []ResponsesTool             `json:"tools"`
	ToolChoice         any                         `json:"tool_choice"`
	Truncation         string                      `json:"truncation"`
	ParallelToolCalls  bool                        `json:"parallel_tool_calls"`
	Text               ResponsesTextField          `json:"text"`
	TopP               float64                     `json:"top_p"`
	PresencePenalty    float64                     `json:"presence_penalty"`
	FrequencyPenalty   float64                     `json:"frequency_penalty"`
	TopLogprobs        int                         `json:"top_logprobs"`
	Temperature        float64                     `json:"temperature"`
	Reasoning          *ResponsesReasoningOutput   `json:"reasoning"`
	Usage              *ResponsesUsage             `json:"usage"`
	MaxOutputTokens    *int                        `json:"max_output_tokens"`
	MaxToolCalls       *int                        `json:"max_tool_calls"`
	Store              bool                        `json:"store"`
	Background         bool                        `json:"background"`
	ServiceTier        string                      `json:"service_tier"`
	Metadata           map[string]any              `json:"metadata"`
	SafetyIdentifier   *string                     `json:"safety_identifier"`
	PromptCacheKey     *string                     `json:"prompt_cache_key"`
}

type ResponsesOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"` // "message", "function_call", or "reasoning"
	Status    string                   `json:"status,omitempty"`
	Role      string                   `json:"role,omitempty"`      // for message
	Content   []ResponsesOutputContent `json:"content,omitempty"`   // for message
	CallID    string                   `json:"call_id,omitempty"`   // for function_call
	Name      string                   `json:"name,omitempty"`      // for function_call
	Arguments string                   `json:"arguments,omitempty"` // for function_call

	// Reasoning fields
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`           // for reasoning
	EncryptedContent string                      `json:"encrypted_content,omitempty"` // for reasoning
}

type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

type ResponsesOutputContent struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
}

type ResponsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponsesUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	OutputTokens        int                          `json:"output_tokens"`
	TotalTokens         int                          `json:"total_tokens"`
	InputTokensDetails  ResponsesInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails ResponsesOutputTokensDetails `json:"output_tokens_details"`
}

// derefFloat64 returns the value of a float64 pointer, or a default if nil.
func derefFloat64(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// ToResponse converts an api.ChatResponse to a Responses API response.
// The request is used to echo back request parameters in the response.
func ToResponse(model, responseID, itemID string, chatResponse api.ChatResponse, request ResponsesRequest) ResponsesResponse {
	var output []ResponsesOutputItem

	// Add reasoning item if thinking is present
	if chatResponse.Message.Thinking != "" {
		output = append(output, ResponsesOutputItem{
			ID:   fmt.Sprintf("rs_%s", responseID),
			Type: "reasoning",
			Summary: []ResponsesReasoningSummary{
				{
					Type: "summary_text",
					Text: chatResponse.Message.Thinking,
				},
			},
			EncryptedContent: chatResponse.Message.Thinking,
		})
	}

	if len(chatResponse.Message.ToolCalls) > 0 {
		toolCalls := ToToolCalls(chatResponse.Message.ToolCalls)
		for i, tc := range toolCalls {
			output = append(output, ResponsesOutputItem{
				ID:        fmt.Sprintf("fc_%s_%d", responseID, i),
				Type:      "function_call",
				Status:    "completed",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	} else {
		output = append(output, ResponsesOutputItem{
			ID:     itemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []ResponsesOutputContent{
				{
					Type:        "output_text",
					Text:        chatResponse.Message.Content,
					Annotations: []any{},
					Logprobs:    []any{},
				},
			},
		})
	}

	var instructions *string
	if request.Instructions != "" {
		instructions = &request.Instructions
	}

	truncation := "disabled"
	if request.Truncation != nil {
		truncation = *request.Truncation
	}

	tools := request.Tools
	if tools == nil {
		tools = []ResponsesTool{}
	}

	text := ResponsesTextField{
		Format: ResponsesTextFormat{Type: "text"},
	}
	if request.Text != nil && request.Text.Format != nil {
		text.Format = *request.Text.Format
	}

	var reasoning *ResponsesReasoningOutput
	if request.Reasoning.Effort != "" || request.Reasoning.Summary != "" {
		reasoning = &ResponsesReasoningOutput{}
		if request.Reasoning.Effort != "" {
			reasoning.Effort = &request.Reasoning.Effort
		}
		if request.Reasoning.Summary != "" {
			reasoning.Summary = &request.Reasoning.Summary
		}
	}

	return ResponsesResponse{
		ID:                 responseID,
		Object:             "response",
		CreatedAt:          chatResponse.CreatedAt.Unix(),
		CompletedAt:        nil,
		Status:             "completed",
		IncompleteDetails:  nil,
		Model:              model,
		PreviousResponseID: nil,
		Instructions:       instructions,
		Output:             output,
		Error:              nil,
		Tools:              tools,
		ToolChoice:         "auto",
		Truncation:         truncation,
		ParallelToolCalls:  true,
		Text:               text,
		TopP:               derefFloat64(request.TopP, 1.0),
		PresencePenalty:    0,
		FrequencyPenalty:   0,
		TopLogprobs:        0,
		Temperature:        derefFloat64(request.Temperature, 1.0),
		Reasoning:          reasoning,
		Usage: &ResponsesUsage{
			InputTokens:        chatResponse.PromptEvalCount,
			OutputTokens:       chatResponse.EvalCount,
			TotalTokens:        chatResponse.PromptEvalCount + chatResponse.EvalCount,
			InputTokensDetails:  ResponsesInputTokensDetails{CachedTokens: 0},
			OutputTokensDetails: ResponsesOutputTokensDetails{ReasoningTokens: 0},
		},
		MaxOutputTokens:  request.MaxOutputTokens,
		MaxToolCalls:     nil,
		Store:            false,
		Background:       request.Background,
		ServiceTier:      "default",
		Metadata:         map[string]any{},
		SafetyIdentifier: nil,
		PromptCacheKey:   nil,
	}
}
