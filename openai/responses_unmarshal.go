// responses_unmarshal.go - JSON Unmarshaling für Responses API Input Types
//
// Dieses Modul enthält:
// - ResponsesInputMessage.UnmarshalJSON
// - unmarshalResponsesInputItem
// - ResponsesInput.UnmarshalJSON

package openai

import (
	"encoding/json"
	"fmt"
)

func (m *ResponsesInputMessage) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	m.Type = aux.Type
	m.Role = aux.Role

	if len(aux.Content) == 0 {
		return nil
	}

	// Try to parse content as a string first (shorthand format)
	var contentStr string
	if err := json.Unmarshal(aux.Content, &contentStr); err == nil {
		m.Content = []ResponsesContent{
			ResponsesTextContent{Type: "input_text", Text: contentStr},
		}
		return nil
	}

	// Otherwise, parse as an array of content items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Content, &rawItems); err != nil {
		return fmt.Errorf("content must be a string or array: %w", err)
	}

	m.Content = make([]ResponsesContent, 0, len(rawItems))
	for i, raw := range rawItems {
		var typeField struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &typeField); err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}

		switch typeField.Type {
		case "input_text":
			var content ResponsesTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "input_image":
			var content ResponsesImageContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "output_text":
			var content ResponsesOutputTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		default:
			return fmt.Errorf("content[%d]: unknown content type: %s", i, typeField.Type)
		}
	}

	return nil
}

// unmarshalResponsesInputItem unmarshals a single input item from JSON.
func unmarshalResponsesInputItem(data []byte) (ResponsesInputItem, error) {
	var typeField struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	// Handle shorthand message format: {"role": "...", "content": "..."}
	itemType := typeField.Type
	if itemType == "" && typeField.Role != "" {
		itemType = "message"
	}

	switch itemType {
	case "message":
		var msg ResponsesInputMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, err
		}
		return msg, nil
	case "function_call":
		var fc ResponsesFunctionCall
		if err := json.Unmarshal(data, &fc); err != nil {
			return nil, err
		}
		return fc, nil
	case "function_call_output":
		var output ResponsesFunctionCallOutput
		if err := json.Unmarshal(data, &output); err != nil {
			return nil, err
		}
		return output, nil
	case "reasoning":
		var reasoning ResponsesReasoningInput
		if err := json.Unmarshal(data, &reasoning); err != nil {
			return nil, err
		}
		return reasoning, nil
	default:
		return nil, fmt.Errorf("unknown input item type: %s", typeField.Type)
	}
}

func (r *ResponsesInput) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		r.Text = s
		return nil
	}

	// Otherwise, try array of input items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(data, &rawItems); err != nil {
		return fmt.Errorf("input must be a string or array: %w", err)
	}

	r.Items = make([]ResponsesInputItem, 0, len(rawItems))
	for i, raw := range rawItems {
		item, err := unmarshalResponsesInputItem(raw)
		if err != nil {
			return fmt.Errorf("input[%d]: %w", i, err)
		}
		r.Items = append(r.Items, item)
	}

	return nil
}
