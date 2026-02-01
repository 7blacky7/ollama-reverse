// Package harmony implementiert den Harmony-Parser fuer Ollama.
//
// Modul: harmony_handler.go - HarmonyMessageHandler
// Enthaelt: HarmonyMessageHandler und Parsing-Zustandsverwaltung
package harmony

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// harmonyMessageState represents the current state of message processing
type harmonyMessageState int

const (
	harmonyMessageState_Normal harmonyMessageState = iota
	harmonyMessageState_Thinking
	harmonyMessageState_ToolCalling
)

// HarmonyMessageHandler processes harmony events and accumulates content appropriately.
// This is a higher level interface that maps harmony concepts into ollama concepts
type HarmonyMessageHandler struct {
	state           harmonyMessageState
	HarmonyParser   *HarmonyParser
	FunctionNameMap *FunctionNameMap
	toolAccumulator *HarmonyToolCallAccumulator
	convertedTools  map[string]struct{}
}

// NewHarmonyMessageHandler creates a new message handler
func NewHarmonyMessageHandler() *HarmonyMessageHandler {
	return &HarmonyMessageHandler{
		state: harmonyMessageState_Normal,
		HarmonyParser: &HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		},
		FunctionNameMap: NewFunctionNameMap(),
		convertedTools:  make(map[string]struct{}),
	}
}

// AddContent processes the content and returns the content, thinking, and tool content.
// content and thinking are already fully parsed, but tool content still needs to be passed to the tool parser
func (h *HarmonyMessageHandler) AddContent(content string, toolParser *HarmonyToolCallAccumulator) (string, string, string) {
	contentSb := strings.Builder{}
	thinkingSb := strings.Builder{}
	toolContentSb := strings.Builder{}

	events := h.HarmonyParser.AddContent(content)
	for _, event := range events {
		switch event := event.(type) {
		case HarmonyEventHeaderComplete:
			logutil.Trace("harmony event header complete", "header", event.Header)
			switch event.Header.Channel {
			case "analysis":
				if event.Header.Recipient != "" {
					h.state = harmonyMessageState_ToolCalling
					// event.Header.Recipient is the tool name, something like
					// "browser.search" for a built-in, or "functions.calc" for a
					// custom one
					toolParser.SetToolName(event.Header.Recipient)
				} else {
					h.state = harmonyMessageState_Thinking
				}
			case "commentary":
				if event.Header.Recipient != "" {
					h.state = harmonyMessageState_ToolCalling
					toolParser.SetToolName(event.Header.Recipient)
				} else {
					h.state = harmonyMessageState_Normal
				}
			case "final":
				h.state = harmonyMessageState_Normal
			}
		case HarmonyEventContentEmitted:
			logutil.Trace("harmony event content", "content", event.Content, "state", h.state)
			if h.state == harmonyMessageState_Normal {
				contentSb.WriteString(event.Content)
			} else if h.state == harmonyMessageState_Thinking {
				thinkingSb.WriteString(event.Content)
			} else if h.state == harmonyMessageState_ToolCalling {
				toolContentSb.WriteString(event.Content)
			}
		case HarmonyEventMessageEnd:
			h.state = harmonyMessageState_Normal
		}
	}
	return contentSb.String(), thinkingSb.String(), toolContentSb.String()
}

func (h *HarmonyMessageHandler) CreateToolParser() *HarmonyToolCallAccumulator {
	return &HarmonyToolCallAccumulator{
		state:           harmonyToolCallState_Normal,
		currentToolName: nil,
	}
}

// Init initializes the handler with tools, optional last message, and think value
// Implements the Parser interface
func (h *HarmonyMessageHandler) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	// Initialize the harmony parser
	if h.HarmonyParser == nil {
		h.HarmonyParser = &HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		}
	}

	// Handle prefill for chat mode
	if lastMessage != nil {
		// Convert api.Message to local Message type
		localMsg := &Message{
			Role:     lastMessage.Role,
			Content:  lastMessage.Content,
			Thinking: lastMessage.Thinking,
		}
		h.HarmonyParser.AddImplicitStartOrPrefill(localMsg)
	} else {
		h.HarmonyParser.AddImplicitStart()
	}

	// Initialize tool accumulator
	h.toolAccumulator = h.CreateToolParser()

	// Process tools and return renamed versions
	if len(tools) == 0 {
		return tools
	}

	processedTools := make([]api.Tool, len(tools))
	copy(processedTools, tools)
	for i, tool := range processedTools {
		if tool.Function.Name != "" {
			processedTools[i].Function.Name = h.FunctionNameMap.ConvertAndAdd(tool.Function.Name)
			h.convertedTools[tool.Function.Name] = struct{}{}
		}
	}
	return processedTools
}

// Add implements the Parser interface - processes streamed content and extracts content, thinking, and tool calls
func (h *HarmonyMessageHandler) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	content, thinking, toolContent := h.AddContent(s, h.toolAccumulator)
	if toolContent != "" {
		h.toolAccumulator.Add(toolContent)
	}

	// tool calls always happen one at a time, and always at the end of a message,
	// so for simplicity we defer parsing them until we know we're done
	if done {
		toolName, raw := h.toolAccumulator.Drain()
		if toolName != nil {
			name := strings.TrimPrefix(*toolName, "functions.")
			name = h.FunctionNameMap.OriginalFromConverted(name)
			var args api.ToolCallFunctionArguments
			if err := json.Unmarshal([]byte(raw), &args); err != nil {
				return "", "", nil, fmt.Errorf("error parsing tool call: raw='%s', err=%w", raw, err)
			}
			calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: args}})
		}
	}

	return content, thinking, calls, nil
}

// HasToolSupport implements the Parser interface
func (h *HarmonyMessageHandler) HasToolSupport() bool {
	return true
}

// HasThinkingSupport implements the Parser interface
func (h *HarmonyMessageHandler) HasThinkingSupport() bool {
	return true
}
