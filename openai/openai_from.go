// openai_from.go - Konvertierungsfunktionen von OpenAI-Format zu API-Format
//
// Enthaelt:
// - FromChatRequest: Chat-Completion Request konvertieren
// - FromCompleteRequest: Text-Completion Request konvertieren
// - FromCompletionToolCall: Tool-Aufrufe konvertieren
// - Hilfsfunktionen: nameFromToolCallID
//
// Verwandte Dateien:
// - openai_types.go: Typdefinitionen
// - openai_to.go: Konvertierung API -> OpenAI Format
// - openai_image.go: Bild-bezogene Konvertierungen
package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

// FromChatRequest konvertiert einen ChatCompletionRequest zu api.ChatRequest
func FromChatRequest(r ChatCompletionRequest) (*api.ChatRequest, error) {
	var messages []api.Message
	for _, msg := range r.Messages {
		toolName := ""
		if strings.ToLower(msg.Role) == "tool" {
			toolName = msg.Name
			if toolName == "" && msg.ToolCallID != "" {
				toolName = nameFromToolCallID(r.Messages, msg.ToolCallID)
			}
		}
		switch content := msg.Content.(type) {
		case string:
			toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
			if err != nil {
				return nil, err
			}
			messages = append(messages, api.Message{Role: msg.Role, Content: content, Thinking: msg.Reasoning, ToolCalls: toolCalls, ToolName: toolName, ToolCallID: msg.ToolCallID})
		case []any:
			for _, c := range content {
				data, ok := c.(map[string]any)
				if !ok {
					return nil, errors.New("invalid message format")
				}
				switch data["type"] {
				case "text":
					text, ok := data["text"].(string)
					if !ok {
						return nil, errors.New("invalid message format")
					}
					messages = append(messages, api.Message{Role: msg.Role, Content: text})
				case "image_url":
					var url string
					if urlMap, ok := data["image_url"].(map[string]any); ok {
						if url, ok = urlMap["url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					} else {
						if url, ok = data["image_url"].(string); !ok {
							return nil, errors.New("invalid message format")
						}
					}

					img, err := decodeImageURL(url)
					if err != nil {
						return nil, err
					}

					messages = append(messages, api.Message{Role: msg.Role, Images: []api.ImageData{img}})
				default:
					return nil, errors.New("invalid message format")
				}
			}
			// Da wir oben mehrere Messages hinzugefuegt haben koennten,
			// fuegen wir Tool-Calls zur letzten Message hinzu
			if len(messages) > 0 && len(msg.ToolCalls) > 0 {
				toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
				if err != nil {
					return nil, err
				}
				messages[len(messages)-1].ToolCalls = toolCalls
				messages[len(messages)-1].ToolName = toolName
				messages[len(messages)-1].ToolCallID = msg.ToolCallID
				messages[len(messages)-1].Thinking = msg.Reasoning
			}
		default:
			// Content ist nur optional wenn Tool-Calls vorhanden sind
			if msg.ToolCalls == nil {
				return nil, fmt.Errorf("invalid message content type: %T", content)
			}

			toolCalls, err := FromCompletionToolCall(msg.ToolCalls)
			if err != nil {
				return nil, err
			}
			messages = append(messages, api.Message{Role: msg.Role, Thinking: msg.Reasoning, ToolCalls: toolCalls, ToolCallID: msg.ToolCallID})
		}
	}

	options := make(map[string]any)

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []any:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			}
		}
		options["stop"] = stops
	}

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed
	}

	if r.FrequencyPenalty != nil {
		options["frequency_penalty"] = *r.FrequencyPenalty
	}

	if r.PresencePenalty != nil {
		options["presence_penalty"] = *r.PresencePenalty
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else {
		options["top_p"] = 1.0
	}

	var format json.RawMessage
	if r.ResponseFormat != nil {
		switch strings.ToLower(strings.TrimSpace(r.ResponseFormat.Type)) {
		// Unterstuetzung fuer alten "json_object" Typ fuer OpenAI-Kompatibilitaet
		case "json_object":
			format = json.RawMessage(`"json"`)
		case "json_schema":
			if r.ResponseFormat.JsonSchema != nil {
				format = r.ResponseFormat.JsonSchema.Schema
			}
		}
	}

	var think *api.ThinkValue
	var effort string

	if r.Reasoning != nil {
		effort = r.Reasoning.Effort
	} else if r.ReasoningEffort != nil {
		effort = *r.ReasoningEffort
	}

	if effort != "" {
		if !slices.Contains([]string{"high", "medium", "low", "none"}, effort) {
			return nil, fmt.Errorf("invalid reasoning value: '%s' (must be \"high\", \"medium\", \"low\", or \"none\")", effort)
		}

		if effort == "none" {
			think = &api.ThinkValue{Value: false}
		} else {
			think = &api.ThinkValue{Value: effort}
		}
	}

	return &api.ChatRequest{
		Model:           r.Model,
		Messages:        messages,
		Format:          format,
		Options:         options,
		Stream:          &r.Stream,
		Tools:           r.Tools,
		Think:           think,
		Logprobs:        r.Logprobs != nil && *r.Logprobs,
		TopLogprobs:     r.TopLogprobs,
		DebugRenderOnly: r.DebugRenderOnly,
	}, nil
}

// nameFromToolCallID findet den Funktionsnamen anhand der ToolCallID
func nameFromToolCallID(messages []Message, toolCallID string) string {
	// Rueckwaerts iterieren um bei doppelten IDs die letzte zu nehmen
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		for _, tc := range msg.ToolCalls {
			if tc.ID == toolCallID {
				return tc.Function.Name
			}
		}
	}
	return ""
}

// FromCompletionToolCall konvertiert OpenAI ToolCall Format zu api.ToolCall
func FromCompletionToolCall(toolCalls []ToolCall) ([]api.ToolCall, error) {
	apiToolCalls := make([]api.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		apiToolCalls[i].ID = tc.ID
		apiToolCalls[i].Function.Name = tc.Function.Name
		err := json.Unmarshal([]byte(tc.Function.Arguments), &apiToolCalls[i].Function.Arguments)
		if err != nil {
			return nil, errors.New("invalid tool call arguments")
		}
	}

	return apiToolCalls, nil
}

// FromCompleteRequest konvertiert einen CompletionRequest zu api.GenerateRequest
func FromCompleteRequest(r CompletionRequest) (api.GenerateRequest, error) {
	options := make(map[string]any)

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []any:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			} else {
				return api.GenerateRequest{}, fmt.Errorf("invalid type for 'stop' field: %T", s)
			}
		}
		options["stop"] = stops
	}

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed
	}

	options["frequency_penalty"] = r.FrequencyPenalty

	options["presence_penalty"] = r.PresencePenalty

	if r.TopP != 0.0 {
		options["top_p"] = r.TopP
	} else {
		options["top_p"] = 1.0
	}

	var logprobs bool
	var topLogprobs int
	if r.Logprobs != nil && *r.Logprobs > 0 {
		logprobs = true
		topLogprobs = *r.Logprobs
	}

	return api.GenerateRequest{
		Model:           r.Model,
		Prompt:          r.Prompt,
		Options:         options,
		Stream:          &r.Stream,
		Suffix:          r.Suffix,
		Logprobs:        logprobs,
		TopLogprobs:     topLogprobs,
		DebugRenderOnly: r.DebugRenderOnly,
	}, nil
}
