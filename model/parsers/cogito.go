// =============================================================================
// Modul: cogito.go
// Beschreibung: Cogito-Parser für Tool-Calls und Thinking-Support.
//               Verwendet spezielle Unicode-Tags für Tool-Call-Erkennung.
//               Die State-Machine (eat) befindet sich in cogito_eat.go.
// =============================================================================

package parsers

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/ollama/ollama/api"
)

// Parser-State-Enum für den Cogito-Parser
type CogitoParserState int

const (
	CogitoCollectingThinking CogitoParserState = iota
	CogitoCollectingContent
	CogitoCollectingToolCalls
	CogitoCollectingToolOutput
)

// Tag-Konstanten für Cogito-spezifische Markup-Erkennung
const (
	cogitoThinkingCloseTag    = "</think>"
	cogitoToolCallsBeginTag   = "<｜tool▁calls▁begin｜>"
	cogitoToolCallsEndTag     = "<｜tool▁calls▁end｜>"
	cogitoToolCallBeginTag    = "<｜tool▁call▁begin｜>"
	cogitoToolCallEndTag      = "<｜tool▁call▁end｜>"
	cogitoToolSepTag          = "<｜tool▁sep｜>"
	cogitoToolOutputBeginTag  = "<｜tool▁output▁begin｜>"
	cogitoToolOutputEndTag    = "<｜tool▁output▁end｜>"
	cogitoToolOutputsBeginTag = "<｜tool▁outputs▁begin｜>"
	cogitoToolOutputsEndTag   = "<｜tool▁outputs▁end｜>"
)

// CogitoParser implementiert das Parser-Interface für Cogito-Modelle.
// Unterstützt sowohl Tool-Calls als auch Thinking-Mode.
type CogitoParser struct {
	state  CogitoParserState
	buffer strings.Builder
}

// HasToolSupport gibt true zurück - Cogito unterstützt Tool-Calls
func (p *CogitoParser) HasToolSupport() bool {
	return true
}

// HasThinkingSupport gibt true zurück - Cogito unterstützt Thinking
func (p *CogitoParser) HasThinkingSupport() bool {
	return true
}

// setInitialState setzt den initialen Parser-State basierend auf Kontext.
// Berücksichtigt: Prefill, Thinking-Einstellung, Tool-Verfügbarkeit.
func (p *CogitoParser) setInitialState(lastMessage *api.Message, tools []api.Tool, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Prüfe sowohl Model-Capability ALS AUCH Request-Preference
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	// thinkingEnabled sollte für Tools auf false gesetzt sein

	if !thinkingEnabled {
		p.state = CogitoCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = CogitoCollectingContent
		return
	}

	// Hinweis: Bei Cogito mit Tools wollen wir nicht denken
	if len(tools) > 0 {
		p.state = CogitoCollectingContent
		return
	}

	p.state = CogitoCollectingThinking
}

// Init initialisiert den Parser mit Tools und optionaler letzter Nachricht
func (p *CogitoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.setInitialState(lastMessage, tools, thinkValue)
	return tools
}

// Event-Interface für typsichere Event-Verarbeitung
type cogitoEvent interface {
	isCogitoEvent()
}

// cogitoEventThinkingContent repräsentiert ein Thinking-Content-Event
type cogitoEventThinkingContent struct {
	content string
}

// cogitoEventContent repräsentiert ein normales Content-Event
type cogitoEventContent struct {
	content string
}

// cogitoEventToolCall repräsentiert ein Tool-Call-Event
type cogitoEventToolCall struct {
	toolCall api.ToolCall
}

// Interface-Implementierungen für Event-Typen
func (cogitoEventThinkingContent) isCogitoEvent() {}
func (cogitoEventContent) isCogitoEvent()         {}
func (cogitoEventToolCall) isCogitoEvent()        {}

// Add verarbeitet eingehenden Stream-Content und gibt Content, Thinking und Tool-Calls zurück
func (p *CogitoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case cogitoEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case cogitoEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case cogitoEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

// parseEvents extrahiert alle verfügbaren Events aus dem Buffer
func (p *CogitoParser) parseEvents() []cogitoEvent {
	var all []cogitoEvent

	keepLooping := true
	for keepLooping {
		var events []cogitoEvent
		// eat() ist in cogito_eat.go implementiert
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// parseToolCallContent parst den Inhalt eines einzelnen Tool-Calls.
// Erwartetes Format: function<｜tool▁sep｜>tool_name\n```json\n{args}\n```
func (p *CogitoParser) parseToolCallContent(content string) (api.ToolCall, error) {
	// Trenne am Separator-Tag
	parts := strings.SplitN(content, cogitoToolSepTag, 2)
	if len(parts) < 2 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	nameAndArgs := parts[1]

	// Finde JSON-Block
	jsonStart := strings.Index(nameAndArgs, "\n```json\n")
	if jsonStart == -1 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	toolName := strings.TrimSpace(nameAndArgs[:jsonStart])
	jsonContent := nameAndArgs[jsonStart+len("\n```json\n"):]

	// Finde Ende des JSON-Blocks
	jsonEnd := strings.Index(jsonContent, "\n```")
	if jsonEnd == -1 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	argsJSON := jsonContent[:jsonEnd]

	// Parse JSON-Argumente
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}
