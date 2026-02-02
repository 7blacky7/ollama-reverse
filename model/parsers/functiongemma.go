// =============================================================================
// Modul: functiongemma.go
// Beschreibung: FunctionGemma-Parser für Tool-Calls im Format
//               <start_function_call>call:name{args}<end_function_call>.
//               Das Wert-Parsing befindet sich in functiongemma_values.go.
// =============================================================================

package parsers

import (
	"regexp"
	"strings"

	"github.com/ollama/ollama/api"
)

// Parser-State-Enum für den FunctionGemma-Parser
type FunctionGemmaParserState int

const (
	FunctionGemmaCollectingContent FunctionGemmaParserState = iota
	FunctionGemmaCollectingToolCalls
)

// Tag-Konstanten für Tool-Call-Erkennung
const (
	functionGemmaFunctionCallOpen  = "<start_function_call>"
	functionGemmaFunctionCallClose = "<end_function_call>"
)

// FunctionGemmaParser implementiert das Parser-Interface für FunctionGemma-Modelle.
// Format: <start_function_call>call:function_name{key:value,key:value}<end_function_call>
type FunctionGemmaParser struct {
	state  FunctionGemmaParserState
	buffer strings.Builder
	tools  []api.Tool
}

// HasToolSupport gibt true zurück - FunctionGemma unterstützt Tool-Calls
func (p *FunctionGemmaParser) HasToolSupport() bool { return true }

// HasThinkingSupport gibt false zurück - FunctionGemma unterstützt kein Thinking
func (p *FunctionGemmaParser) HasThinkingSupport() bool { return false }

// Init initialisiert den Parser mit Tools und optionaler letzter Nachricht
func (p *FunctionGemmaParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.state = FunctionGemmaCollectingContent
	return tools
}

// Event-Interface für typsichere Event-Verarbeitung
type functionGemmaEvent interface {
	isFunctionGemmaEvent()
}

// FunctionGemmaEventContent repräsentiert ein Content-Event
type FunctionGemmaEventContent struct {
	content string
}

// functionGemmaEventToolCall repräsentiert ein Tool-Call-Event
type functionGemmaEventToolCall struct {
	toolCall api.ToolCall
}

// Interface-Implementierungen für Event-Typen
func (FunctionGemmaEventContent) isFunctionGemmaEvent()  {}
func (functionGemmaEventToolCall) isFunctionGemmaEvent() {}

// Add verarbeitet eingehenden Stream-Content und gibt Content und Tool-Calls zurück
func (p *FunctionGemmaParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case functionGemmaEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case FunctionGemmaEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), "", toolCalls, nil
}

// parseEvents extrahiert alle verfügbaren Events aus dem Buffer
func (p *FunctionGemmaParser) parseEvents() []functionGemmaEvent {
	var all []functionGemmaEvent

	keepLooping := true
	for keepLooping {
		var events []functionGemmaEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// emitWithPartialCheck extrahiert eindeutigen Content vor einem möglichen partiellen Tag
func (p *FunctionGemmaParser) emitWithPartialCheck(bufStr, tag string) (unambiguous, ambiguous string) {
	if overlapLen := overlap(bufStr, tag); overlapLen > 0 {
		beforePartialTag := bufStr[:len(bufStr)-overlapLen]
		return beforePartialTag, bufStr[len(beforePartialTag):]
	}
	return bufStr, ""
}

// eat verarbeitet den Buffer und extrahiert Events basierend auf dem aktuellen State
func (p *FunctionGemmaParser) eat() ([]functionGemmaEvent, bool) {
	bufStr := p.buffer.String()
	if bufStr == "" {
		return nil, false
	}

	switch p.state {
	case FunctionGemmaCollectingContent:
		// Suche nach vollständigem Open-Tag
		if strings.Contains(bufStr, functionGemmaFunctionCallOpen) {
			split := strings.SplitN(bufStr, functionGemmaFunctionCallOpen, 2)
			content := split[0]
			p.buffer.Reset()
			p.buffer.WriteString(split[1])
			p.state = FunctionGemmaCollectingToolCalls
			if content != "" {
				return []functionGemmaEvent{FunctionGemmaEventContent{content: content}}, true
			}
			return nil, true
		}
		// Prüfe auf partiellen Open-Tag
		unambig, ambig := p.emitWithPartialCheck(bufStr, functionGemmaFunctionCallOpen)
		p.buffer.Reset()
		p.buffer.WriteString(ambig)
		if unambig != "" {
			return []functionGemmaEvent{FunctionGemmaEventContent{content: unambig}}, false
		}
		return nil, false

	case FunctionGemmaCollectingToolCalls:
		// Suche nach vollständigem Close-Tag
		if strings.Contains(bufStr, functionGemmaFunctionCallClose) {
			split := strings.SplitN(bufStr, functionGemmaFunctionCallClose, 2)
			remaining := split[1]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)

			var events []functionGemmaEvent
			if tc, err := p.parseToolCall(split[0]); err == nil {
				events = append(events, functionGemmaEventToolCall{toolCall: tc})
			}

			// Prüfe ob weitere Tool-Calls folgen
			if !strings.Contains(remaining, functionGemmaFunctionCallOpen) {
				p.state = FunctionGemmaCollectingContent
			}
			return events, true
		}
		return nil, false
	}

	return nil, false
}

// Regex für Tool-Call-Parsing: call:function_name{args}
var functionGemmaCallRegex = regexp.MustCompile(`call:([^{]+)\{(.*)\}`)

// parseToolCall parst einen einzelnen Tool-Call aus dem Content
func (p *FunctionGemmaParser) parseToolCall(content string) (api.ToolCall, error) {
	toolCall := api.ToolCall{}

	// Extrahiere Funktionsnamen und Argumente
	match := functionGemmaCallRegex.FindStringSubmatch(content)
	if len(match) < 3 {
		return toolCall, nil
	}

	toolCall.Function.Name = match[1]
	argsStr := match[2]

	// Argumente parsen (implementiert in functiongemma_values.go)
	toolCall.Function.Arguments = p.parseArguments(argsStr)

	return toolCall, nil
}
