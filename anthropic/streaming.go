// streaming.go
// Streaming-Modul: StreamConverter und Event-Verarbeitung fuer SSE-Responses

package anthropic

import (
	"encoding/json"
	"log/slog"

	"github.com/ollama/ollama/api"
)

// StreamConverter manages state for converting Ollama streaming responses to Anthropic format
type StreamConverter struct {
	ID              string
	Model           string
	firstWrite      bool
	contentIndex    int
	inputTokens     int
	outputTokens    int
	thinkingStarted bool
	thinkingDone    bool
	textStarted     bool
	toolCallsSent   map[string]bool
}

func NewStreamConverter(id, model string) *StreamConverter {
	return &StreamConverter{
		ID:            id,
		Model:         model,
		firstWrite:    true,
		toolCallsSent: make(map[string]bool),
	}
}

// StreamEvent represents a streaming event to be sent to the client
type StreamEvent struct {
	Event string
	Data  any
}

// Process converts an Ollama ChatResponse to Anthropic streaming events
func (c *StreamConverter) Process(r api.ChatResponse) []StreamEvent {
	var events []StreamEvent

	if c.firstWrite {
		c.firstWrite = false
		c.inputTokens = r.Metrics.PromptEvalCount

		events = append(events, StreamEvent{
			Event: "message_start",
			Data: MessageStartEvent{
				Type: "message_start",
				Message: MessagesResponse{
					ID:      c.ID,
					Type:    "message",
					Role:    "assistant",
					Model:   c.Model,
					Content: []ContentBlock{},
					Usage: Usage{
						InputTokens:  c.inputTokens,
						OutputTokens: 0,
					},
				},
			},
		})
	}

	if r.Message.Thinking != "" && !c.thinkingDone {
		if !c.thinkingStarted {
			c.thinkingStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type:     "thinking",
						Thinking: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:     "thinking_delta",
					Thinking: r.Message.Thinking,
				},
			},
		})
	}

	if r.Message.Content != "" {
		if c.thinkingStarted && !c.thinkingDone {
			c.thinkingDone = true
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
		}

		if !c.textStarted {
			c.textStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type: "text",
						Text: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type: "text_delta",
					Text: r.Message.Content,
				},
			},
		})
	}

	for _, tc := range r.Message.ToolCalls {
		if c.toolCallsSent[tc.ID] {
			continue
		}

		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
			c.textStarted = false
		}

		argsJSON, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("failed to marshal tool arguments", "error", err, "tool_id", tc.ID)
			continue
		}

		events = append(events, StreamEvent{
			Event: "content_block_start",
			Data: ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: c.contentIndex,
				ContentBlock: ContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: map[string]any{},
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:        "input_json_delta",
					PartialJSON: string(argsJSON),
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})

		c.toolCallsSent[tc.ID] = true
		c.contentIndex++
	}

	if r.Done {
		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		} else if c.thinkingStarted && !c.thinkingDone {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		}

		c.outputTokens = r.Metrics.EvalCount
		stopReason := mapStopReason(r.DoneReason, len(c.toolCallsSent) > 0)

		events = append(events, StreamEvent{
			Event: "message_delta",
			Data: MessageDeltaEvent{
				Type: "message_delta",
				Delta: MessageDelta{
					StopReason: stopReason,
				},
				Usage: DeltaUsage{
					OutputTokens: c.outputTokens,
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "message_stop",
			Data: MessageStopEvent{
				Type: "message_stop",
			},
		})
	}

	return events
}
