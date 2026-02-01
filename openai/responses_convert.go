// responses_convert.go - Konvertierungsfunktionen für Responses API
//
// Dieses Modul enthält Funktionen zur Konvertierung zwischen:
// - ResponsesRequest → api.ChatRequest (FromResponsesRequest)
// - ResponsesTool → api.Tool (convertTool)
// - ResponsesInputMessage → api.Message (convertInputMessage)

package openai

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/api"
)

// FromResponsesRequest converts a ResponsesRequest to api.ChatRequest
func FromResponsesRequest(r ResponsesRequest) (*api.ChatRequest, error) {
	var messages []api.Message

	// Add instructions as system message if present
	if r.Instructions != "" {
		messages = append(messages, api.Message{
			Role:    "system",
			Content: r.Instructions,
		})
	}

	// Handle simple string input
	if r.Input.Text != "" {
		messages = append(messages, api.Message{
			Role:    "user",
			Content: r.Input.Text,
		})
	}

	// Handle array of input items
	// Track pending reasoning to merge with the next assistant message
	var pendingThinking string

	for _, item := range r.Input.Items {
		switch v := item.(type) {
		case ResponsesReasoningInput:
			// Store thinking to merge with the next assistant message
			pendingThinking = v.EncryptedContent
		case ResponsesInputMessage:
			msg, err := convertInputMessage(v)
			if err != nil {
				return nil, err
			}
			// If this is an assistant message, attach pending thinking
			if msg.Role == "assistant" && pendingThinking != "" {
				msg.Thinking = pendingThinking
				pendingThinking = ""
			}
			messages = append(messages, msg)
		case ResponsesFunctionCall:
			// Convert function call to assistant message with tool calls
			var args api.ToolCallFunctionArguments
			if v.Arguments != "" {
				if err := json.Unmarshal([]byte(v.Arguments), &args); err != nil {
					return nil, fmt.Errorf("failed to parse function call arguments: %w", err)
				}
			}
			toolCall := api.ToolCall{
				ID: v.CallID,
				Function: api.ToolCallFunction{
					Name:      v.Name,
					Arguments: args,
				},
			}

			// Merge tool call into existing assistant message if it has content or tool calls
			if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
				lastMsg := &messages[len(messages)-1]
				lastMsg.ToolCalls = append(lastMsg.ToolCalls, toolCall)
				if pendingThinking != "" {
					lastMsg.Thinking = pendingThinking
					pendingThinking = ""
				}
			} else {
				msg := api.Message{
					Role:      "assistant",
					ToolCalls: []api.ToolCall{toolCall},
				}
				if pendingThinking != "" {
					msg.Thinking = pendingThinking
					pendingThinking = ""
				}
				messages = append(messages, msg)
			}
		case ResponsesFunctionCallOutput:
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    v.Output,
				ToolCallID: v.CallID,
			})
		}
	}

	// If there's trailing reasoning without a following message, emit it
	if pendingThinking != "" {
		messages = append(messages, api.Message{
			Role:     "assistant",
			Thinking: pendingThinking,
		})
	}

	options := make(map[string]any)

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else { //nolint:staticcheck // SA9003: empty branch
		// TODO(drifkin): OpenAI defaults to 1.0 here, but we don't follow that here
		// in case the model has a different default. It would be best if we
		// understood whether there was a model-specific default and if not, we
		// should also default to 1.0, but that will require some additional
		// plumbing
	}

	if r.MaxOutputTokens != nil {
		options["num_predict"] = *r.MaxOutputTokens
	}

	// Convert tools from Responses API format to api.Tool format
	var tools []api.Tool
	for _, t := range r.Tools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}

	// Handle text format (e.g. json_schema)
	var format json.RawMessage
	if r.Text != nil && r.Text.Format != nil {
		switch r.Text.Format.Type {
		case "json_schema":
			if r.Text.Format.Schema != nil {
				format = r.Text.Format.Schema
			}
		}
	}

	return &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Tools:    tools,
		Format:   format,
	}, nil
}

func convertTool(t ResponsesTool) (api.Tool, error) {
	// Convert parameters from map[string]any to api.ToolFunctionParameters
	var params api.ToolFunctionParameters
	if t.Parameters != nil {
		// Marshal and unmarshal to convert
		b, err := json.Marshal(t.Parameters)
		if err != nil {
			return api.Tool{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return api.Tool{}, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
	}

	var description string
	if t.Description != nil {
		description = *t.Description
	}

	return api.Tool{
		Type: t.Type,
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: description,
			Parameters:  params,
		},
	}, nil
}

func convertInputMessage(m ResponsesInputMessage) (api.Message, error) {
	var content string
	var images []api.ImageData

	for _, c := range m.Content {
		switch v := c.(type) {
		case ResponsesTextContent:
			content += v.Text
		case ResponsesOutputTextContent:
			content += v.Text
		case ResponsesImageContent:
			if v.ImageURL == "" {
				continue // Skip if no URL (FileID not supported)
			}
			img, err := decodeImageURL(v.ImageURL)
			if err != nil {
				return api.Message{}, err
			}
			images = append(images, img)
		}
	}

	return api.Message{
		Role:    m.Role,
		Content: content,
		Images:  images,
	}, nil
}
