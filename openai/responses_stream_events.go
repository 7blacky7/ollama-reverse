// responses_stream_events.go - Stream Event-Handler für Tool Calls und Completion
//
// Dieses Modul enthält die komplexeren Event-Handler:
// - processToolCalls
// - processCompletion
// - buildFinalOutput

package openai

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ollama/ollama/api"
)

func (c *ResponsesStreamConverter) processToolCalls(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	converted := ToToolCalls(toolCalls)

	for i, tc := range converted {
		fcItemID := fmt.Sprintf("fc_%d_%d", rand.Intn(999999), i)

		// Store for final output (with status: completed)
		toolCallItem := map[string]any{
			"id":        fcItemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		}
		c.toolCallItems = append(c.toolCallItems, toolCallItem)

		// response.output_item.added for function call
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": "",
			},
		}))

		// response.function_call_arguments.delta
		if tc.Function.Arguments != "" {
			events = append(events, c.newEvent("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcItemID,
				"output_index": c.outputIndex + i,
				"delta":        tc.Function.Arguments,
			}))
		}

		// response.function_call_arguments.done
		events = append(events, c.newEvent("response.function_call_arguments.done", map[string]any{
			"item_id":      fcItemID,
			"output_index": c.outputIndex + i,
			"arguments":    tc.Function.Arguments,
		}))

		// response.output_item.done for function call
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
			},
		}))
	}

	return events
}

func (c *ResponsesStreamConverter) buildFinalOutput() []any {
	var output []any

	// Add reasoning item if present
	if c.reasoningStarted {
		output = append(output, map[string]any{
			"id":                c.reasoningItemID,
			"type":              "reasoning",
			"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
			"encrypted_content": c.accumulatedThinking,
		})
	}

	// Add tool calls if present
	if len(c.toolCallItems) > 0 {
		for _, item := range c.toolCallItems {
			output = append(output, item)
		}
	} else if c.contentStarted {
		// Add message item if we had text content
		output = append(output, map[string]any{
			"id":     c.itemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			}},
		})
	}

	return output
}

func (c *ResponsesStreamConverter) processCompletion(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning if not done
	events = append(events, c.finishReasoning()...)

	// Emit text completion events if we had text content
	if !c.toolCallsSent && c.contentStarted {
		// response.output_text.done
		events = append(events, c.newEvent("response.output_text.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"text":          c.accumulatedText,
			"logprobs":      []any{},
		}))

		// response.content_part.done
		events = append(events, c.newEvent("response.content_part.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}))

		// response.output_item.done
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":     c.itemID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
				"content": []map[string]any{{
					"type":        "output_text",
					"text":        c.accumulatedText,
					"annotations": []any{},
					"logprobs":    []any{},
				}},
			},
		}))
	}

	// response.completed
	usage := map[string]any{
		"input_tokens":  r.PromptEvalCount,
		"output_tokens": r.EvalCount,
		"total_tokens":  r.PromptEvalCount + r.EvalCount,
		"input_tokens_details": map[string]any{
			"cached_tokens": 0,
		},
		"output_tokens_details": map[string]any{
			"reasoning_tokens": 0,
		},
	}
	response := c.buildResponseObject("completed", c.buildFinalOutput(), usage)
	response["completed_at"] = time.Now().Unix()
	events = append(events, c.newEvent("response.completed", map[string]any{
		"response": response,
	}))

	return events
}
