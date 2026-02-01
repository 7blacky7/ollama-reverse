// responses_stream.go - Streaming-Logik für Responses API
//
// Dieses Modul enthält:
// - ResponsesStreamEvent
// - ResponsesStreamConverter (Stateful Converter für Streaming)
// - Basis Stream-Event-Methoden (Process, Thinking, Text Content)
//
// Siehe responses_stream_events.go für Tool Calls und Completion Events

package openai

import (
	"fmt"
	"math/rand"

	"github.com/ollama/ollama/api"
)

// ResponsesStreamEvent represents a single Server-Sent Event for the Responses API.
type ResponsesStreamEvent struct {
	Event string
	Data  any
}

// ResponsesStreamConverter converts api.ChatResponse objects to Responses API streaming events.
type ResponsesStreamConverter struct {
	responseID string
	itemID     string
	model      string
	request    ResponsesRequest

	firstWrite      bool
	outputIndex     int
	contentIndex    int
	contentStarted  bool
	toolCallsSent   bool
	accumulatedText string
	sequenceNumber  int

	accumulatedThinking string
	reasoningItemID     string
	reasoningStarted    bool
	reasoningDone       bool

	toolCallItems []map[string]any
}

// NewResponsesStreamConverter creates a new converter with the given configuration.
func NewResponsesStreamConverter(responseID, itemID, model string, request ResponsesRequest) *ResponsesStreamConverter {
	return &ResponsesStreamConverter{
		responseID: responseID,
		itemID:     itemID,
		model:      model,
		request:    request,
		firstWrite: true,
	}
}

func (c *ResponsesStreamConverter) newEvent(eventType string, data map[string]any) ResponsesStreamEvent {
	data["type"] = eventType
	data["sequence_number"] = c.sequenceNumber
	c.sequenceNumber++
	return ResponsesStreamEvent{Event: eventType, Data: data}
}

// Process takes a ChatResponse and returns the events that should be emitted.
func (c *ResponsesStreamConverter) Process(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	hasToolCalls := len(r.Message.ToolCalls) > 0
	hasThinking := r.Message.Thinking != ""

	if c.firstWrite {
		c.firstWrite = false
		events = append(events, c.createResponseCreatedEvent())
		events = append(events, c.createResponseInProgressEvent())
	}

	if hasThinking {
		events = append(events, c.processThinking(r.Message.Thinking)...)
	}

	if hasToolCalls {
		events = append(events, c.processToolCalls(r.Message.ToolCalls)...)
		c.toolCallsSent = true
	}

	if !hasToolCalls && !c.toolCallsSent && r.Message.Content != "" {
		events = append(events, c.processTextContent(r.Message.Content)...)
	}

	if r.Done {
		events = append(events, c.processCompletion(r)...)
	}

	return events
}

func (c *ResponsesStreamConverter) createResponseCreatedEvent() ResponsesStreamEvent {
	return c.newEvent("response.created", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) createResponseInProgressEvent() ResponsesStreamEvent {
	return c.newEvent("response.in_progress", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) processThinking(thinking string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	if !c.reasoningStarted {
		c.reasoningStarted = true
		c.reasoningItemID = fmt.Sprintf("rs_%d", rand.Intn(999999))

		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item":         map[string]any{"id": c.reasoningItemID, "type": "reasoning", "summary": []any{}},
		}))
	}

	c.accumulatedThinking += thinking

	events = append(events, c.newEvent("response.reasoning_summary_text.delta", map[string]any{
		"item_id": c.reasoningItemID, "output_index": c.outputIndex, "summary_index": 0, "delta": thinking,
	}))

	return events
}

func (c *ResponsesStreamConverter) finishReasoning() []ResponsesStreamEvent {
	if !c.reasoningStarted || c.reasoningDone {
		return nil
	}
	c.reasoningDone = true

	events := []ResponsesStreamEvent{
		c.newEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id": c.reasoningItemID, "output_index": c.outputIndex, "summary_index": 0, "text": c.accumulatedThinking,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id": c.reasoningItemID, "type": "reasoning",
				"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
				"encrypted_content": c.accumulatedThinking,
			},
		}),
	}

	c.outputIndex++
	return events
}

func (c *ResponsesStreamConverter) processTextContent(content string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	events = append(events, c.finishReasoning()...)

	if !c.contentStarted {
		c.contentStarted = true

		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item":         map[string]any{"id": c.itemID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}},
		}))

		events = append(events, c.newEvent("response.content_part.added", map[string]any{
			"item_id": c.itemID, "output_index": c.outputIndex, "content_index": c.contentIndex,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}},
		}))
	}

	c.accumulatedText += content

	events = append(events, c.newEvent("response.output_text.delta", map[string]any{
		"item_id": c.itemID, "output_index": c.outputIndex, "content_index": 0, "delta": content, "logprobs": []any{},
	}))

	return events
}
