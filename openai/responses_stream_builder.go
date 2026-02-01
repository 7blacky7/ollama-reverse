// responses_stream_builder.go - Response Object Builder für Streaming
//
// Dieses Modul enthält:
// - buildResponseObject (erstellt vollständige Response-Objekte für Streaming-Events)

package openai

import "time"

// buildResponseObject creates a full response object with all required fields for streaming events.
func (c *ResponsesStreamConverter) buildResponseObject(status string, output []any, usage map[string]any) map[string]any {
	var instructions any = nil
	if c.request.Instructions != "" {
		instructions = c.request.Instructions
	}

	truncation := "disabled"
	if c.request.Truncation != nil {
		truncation = *c.request.Truncation
	}

	var tools []any
	if c.request.Tools != nil {
		for _, t := range c.request.Tools {
			tools = append(tools, map[string]any{
				"type": t.Type, "name": t.Name, "description": t.Description,
				"strict": t.Strict, "parameters": t.Parameters,
			})
		}
	}
	if tools == nil {
		tools = []any{}
	}

	textFormat := map[string]any{"type": "text"}
	if c.request.Text != nil && c.request.Text.Format != nil {
		textFormat = map[string]any{"type": c.request.Text.Format.Type}
		if c.request.Text.Format.Name != "" {
			textFormat["name"] = c.request.Text.Format.Name
		}
		if c.request.Text.Format.Schema != nil {
			textFormat["schema"] = c.request.Text.Format.Schema
		}
		if c.request.Text.Format.Strict != nil {
			textFormat["strict"] = *c.request.Text.Format.Strict
		}
	}

	var reasoning any = nil
	if c.request.Reasoning.Effort != "" || c.request.Reasoning.Summary != "" {
		r := map[string]any{}
		if c.request.Reasoning.Effort != "" {
			r["effort"] = c.request.Reasoning.Effort
		} else {
			r["effort"] = nil
		}
		if c.request.Reasoning.Summary != "" {
			r["summary"] = c.request.Reasoning.Summary
		} else {
			r["summary"] = nil
		}
		reasoning = r
	}

	topP := 1.0
	if c.request.TopP != nil {
		topP = *c.request.TopP
	}
	temperature := 1.0
	if c.request.Temperature != nil {
		temperature = *c.request.Temperature
	}

	return map[string]any{
		"id": c.responseID, "object": "response", "created_at": time.Now().Unix(),
		"completed_at": nil, "status": status, "incomplete_details": nil,
		"model": c.model, "previous_response_id": nil, "instructions": instructions,
		"output": output, "error": nil, "tools": tools, "tool_choice": "auto",
		"truncation": truncation, "parallel_tool_calls": true,
		"text": map[string]any{"format": textFormat},
		"top_p": topP, "presence_penalty": 0, "frequency_penalty": 0, "top_logprobs": 0,
		"temperature": temperature, "reasoning": reasoning, "usage": usage,
		"max_output_tokens": c.request.MaxOutputTokens, "max_tool_calls": nil,
		"store": false, "background": c.request.Background, "service_tier": "default",
		"metadata": map[string]any{}, "safety_identifier": nil, "prompt_cache_key": nil,
	}
}
