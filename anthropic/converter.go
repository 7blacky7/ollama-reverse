// converter.go
// Konverter-Modul: Umwandlung zwischen Anthropic- und Ollama-Formaten

package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// FromMessagesRequest converts an Anthropic MessagesRequest to an Ollama api.ChatRequest
func FromMessagesRequest(r MessagesRequest) (*api.ChatRequest, error) {
	var messages []api.Message

	if r.System != nil {
		switch sys := r.System.(type) {
		case string:
			if sys != "" {
				messages = append(messages, api.Message{Role: "system", Content: sys})
			}
		case []any:
			// System can be an array of content blocks
			var content strings.Builder
			for _, block := range sys {
				if blockMap, ok := block.(map[string]any); ok {
					if blockMap["type"] == "text" {
						if text, ok := blockMap["text"].(string); ok {
							content.WriteString(text)
						}
					}
				}
			}
			if content.Len() > 0 {
				messages = append(messages, api.Message{Role: "system", Content: content.String()})
			}
		}
	}

	for _, msg := range r.Messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	options := make(map[string]any)

	options["num_predict"] = r.MaxTokens

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	}

	if r.TopK != nil {
		options["top_k"] = *r.TopK
	}

	if len(r.StopSequences) > 0 {
		options["stop"] = r.StopSequences
	}

	var tools api.Tools
	for _, t := range r.Tools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}

	var think *api.ThinkValue
	if r.Thinking != nil && r.Thinking.Type == "enabled" {
		think = &api.ThinkValue{Value: true}
	}

	stream := r.Stream

	return &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Stream:   &stream,
		Tools:    tools,
		Think:    think,
	}, nil
}

// convertMessage converts an Anthropic MessageParam to Ollama api.Message(s)
func convertMessage(msg MessageParam) ([]api.Message, error) {
	var messages []api.Message
	role := strings.ToLower(msg.Role)

	switch content := msg.Content.(type) {
	case string:
		messages = append(messages, api.Message{Role: role, Content: content})

	case []any:
		var textContent strings.Builder
		var images []api.ImageData
		var toolCalls []api.ToolCall
		var thinking string
		var toolResults []api.Message

		for _, block := range content {
			blockMap, ok := block.(map[string]any)
			if !ok {
				return nil, errors.New("invalid content block format")
			}

			blockType, _ := blockMap["type"].(string)

			switch blockType {
			case "text":
				if text, ok := blockMap["text"].(string); ok {
					textContent.WriteString(text)
				}

			case "image":
				source, ok := blockMap["source"].(map[string]any)
				if !ok {
					return nil, errors.New("invalid image source")
				}

				sourceType, _ := source["type"].(string)
				if sourceType == "base64" {
					data, _ := source["data"].(string)
					decoded, err := base64.StdEncoding.DecodeString(data)
					if err != nil {
						return nil, fmt.Errorf("invalid base64 image data: %w", err)
					}
					images = append(images, decoded)
				} else {
					return nil, fmt.Errorf("invalid image source type: %s. Only base64 images are supported.", sourceType)
				}
				// URL images would need to be fetched - skip for now

			case "tool_use":
				id, ok := blockMap["id"].(string)
				if !ok {
					return nil, errors.New("tool_use block missing required 'id' field")
				}
				name, ok := blockMap["name"].(string)
				if !ok {
					return nil, errors.New("tool_use block missing required 'name' field")
				}
				tc := api.ToolCall{
					ID: id,
					Function: api.ToolCallFunction{
						Name: name,
					},
				}
				if input, ok := blockMap["input"].(map[string]any); ok {
					tc.Function.Arguments = mapToArgs(input)
				}
				toolCalls = append(toolCalls, tc)

			case "tool_result":
				toolUseID, _ := blockMap["tool_use_id"].(string)
				var resultContent string

				switch c := blockMap["content"].(type) {
				case string:
					resultContent = c
				case []any:
					for _, cb := range c {
						if cbMap, ok := cb.(map[string]any); ok {
							if cbMap["type"] == "text" {
								if text, ok := cbMap["text"].(string); ok {
									resultContent += text
								}
							}
						}
					}
				}

				toolResults = append(toolResults, api.Message{
					Role:       "tool",
					Content:    resultContent,
					ToolCallID: toolUseID,
				})

			case "thinking":
				if t, ok := blockMap["thinking"].(string); ok {
					thinking = t
				}
			}
		}

		if textContent.Len() > 0 || len(images) > 0 || len(toolCalls) > 0 || thinking != "" {
			m := api.Message{
				Role:      role,
				Content:   textContent.String(),
				Images:    images,
				ToolCalls: toolCalls,
				Thinking:  thinking,
			}
			messages = append(messages, m)
		}

		// Add tool results as separate messages
		messages = append(messages, toolResults...)

	default:
		return nil, fmt.Errorf("invalid message content type: %T", content)
	}

	return messages, nil
}

// convertTool converts an Anthropic Tool to an Ollama api.Tool
func convertTool(t Tool) (api.Tool, error) {
	var params api.ToolFunctionParameters
	if len(t.InputSchema) > 0 {
		if err := json.Unmarshal(t.InputSchema, &params); err != nil {
			return api.Tool{}, fmt.Errorf("invalid input_schema for tool %q: %w", t.Name, err)
		}
	}

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		},
	}, nil
}

// ToMessagesResponse converts an Ollama api.ChatResponse to an Anthropic MessagesResponse
func ToMessagesResponse(id string, r api.ChatResponse) MessagesResponse {
	var content []ContentBlock

	if r.Message.Thinking != "" {
		content = append(content, ContentBlock{
			Type:     "thinking",
			Thinking: ptr(r.Message.Thinking),
		})
	}

	if r.Message.Content != "" {
		content = append(content, ContentBlock{
			Type: "text",
			Text: ptr(r.Message.Content),
		})
	}

	for _, tc := range r.Message.ToolCalls {
		content = append(content, ContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: tc.Function.Arguments,
		})
	}

	stopReason := mapStopReason(r.DoneReason, len(r.Message.ToolCalls) > 0)

	return MessagesResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      r.Model,
		Content:    content,
		StopReason: stopReason,
		Usage: Usage{
			InputTokens:  r.Metrics.PromptEvalCount,
			OutputTokens: r.Metrics.EvalCount,
		},
	}
}

// mapStopReason converts Ollama done_reason to Anthropic stop_reason
func mapStopReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}

	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		if reason != "" {
			return "stop_sequence"
		}
		return ""
	}
}
