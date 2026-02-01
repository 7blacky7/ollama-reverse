// Package parsers - Ministral Parser Hauptmodul
//
// Diese Datei enthält den MinistralParser für das Parsen von
// Ministral-Modell-Ausgaben mit Unterstützung für:
// - Content-Streaming
// - Thinking-Tags ([THINK]...[/THINK])
// - Tool-Calls ([TOOL_CALLS]...[ARGS]...)
//
// Haupttypen:
// - MinistralParser: Haupt-Parser mit State-Machine
// - ministralEvent*: Event-Typen für verschiedene Output-Arten
package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// Parser-Zustände
type ministralParserState int

const (
	ministralCollectingContent = iota
	ministralCollectingThinkingContent
	ministralCollectingToolName
	ministralCollectingToolArgs
)

// Tag-Konstanten für Ministral-Format
const (
	ministralToolCallsTag = "[TOOL_CALLS]"
	ministralThinkTag     = "[THINK]"
	ministralThinkEndTag  = "[/THINK]"
	ministralArgsTag      = "[ARGS]"
)

// ministralEvent ist das Interface für Parser-Events
type ministralEvent interface {
	isMinistralEvent()
}

// ministralEventContent enthält normalen Content
type ministralEventContent struct {
	content string
}

// ministralEventThinking enthält Thinking-Content
type ministralEventThinking struct {
	thinking string
}

// ministralEventToolCall enthält einen Tool-Aufruf
type ministralEventToolCall struct {
	name string
	args string // Raw JSON String
}

func (ministralEventContent) isMinistralEvent()  {}
func (ministralEventThinking) isMinistralEvent() {}
func (ministralEventToolCall) isMinistralEvent() {}

// MinistralParser parst Ministral-Modell-Ausgaben
type MinistralParser struct {
	state              ministralParserState
	buffer             strings.Builder
	tools              []api.Tool
	hasThinkingSupport bool
	pendingToolName    string // Speichert Tool-Namen während Arg-Sammlung
}

// HasToolSupport gibt zurück ob der Parser Tools unterstützt
func (p *MinistralParser) HasToolSupport() bool {
	return true
}

// HasThinkingSupport gibt zurück ob der Parser Thinking unterstützt
func (p *MinistralParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

// setInitialState setzt den initialen Parser-Zustand
func (p *MinistralParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = ministralCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = ministralCollectingContent
		return
	}

	p.state = ministralCollectingThinkingContent
}

// Init initialisiert den Parser mit Tools und Kontext
func (p *MinistralParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

// toolByName findet ein Tool anhand seines Namens
func toolByName(tools []api.Tool, n string) (*api.Tool, error) {
	for i := range tools {
		if tools[i].Function.Name == n {
			return &tools[i], nil
		}
	}
	return nil, fmt.Errorf("tool '%s' not found", n)
}

// Add fügt neuen Text zum Parser hinzu und gibt Events zurück
func (p *MinistralParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	events := p.parseEvents()

	var contentBuilder, thinkingBuilder strings.Builder
	var toolCalls []api.ToolCall

	for _, event := range events {
		switch e := event.(type) {
		case ministralEventContent:
			contentBuilder.WriteString(e.content)
		case ministralEventThinking:
			thinkingBuilder.WriteString(e.thinking)
		case ministralEventToolCall:
			// Tool validieren
			tool, toolErr := toolByName(p.tools, e.name)
			if toolErr != nil {
				return contentBuilder.String(), thinkingBuilder.String(), toolCalls, toolErr
			}
			// JSON-Argumente parsen
			var args api.ToolCallFunctionArguments
			if jsonErr := json.Unmarshal([]byte(e.args), &args); jsonErr != nil {
				return contentBuilder.String(), thinkingBuilder.String(), toolCalls, jsonErr
			}
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      tool.Function.Name,
					Arguments: args,
				},
			})
		}
	}

	return contentBuilder.String(), thinkingBuilder.String(), toolCalls, nil
}

// parseEvents ruft eat() in einer Schleife auf bis false zurückgegeben wird
func (p *MinistralParser) parseEvents() []ministralEvent {
	var all []ministralEvent
	keepLooping := true
	for keepLooping {
		var events []ministralEvent
		events, keepLooping = p.eat()
		all = append(all, events...)
	}
	return all
}
