package parsers

import (
	"context"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type olmo3ParserState int

const (
	olmo3StateContent olmo3ParserState = iota
	olmo3StateToolCalls
	olmo3StateToolCallsDone
)

const (
	olmo3FuncCallsOpenTag  = "<function_calls>"
	olmo3FuncCallsCloseTag = "</function_calls>"
)

type Olmo3Parser struct {
	state  olmo3ParserState
	buffer strings.Builder
}

func (p *Olmo3Parser) HasToolSupport() bool {
	return true
}

func (p *Olmo3Parser) HasThinkingSupport() bool {
	return false
}

func (p *Olmo3Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = olmo3StateContent
	return tools
}

type olmo3ParserEvent interface {
	isOlmo3ParserEvent()
}

type olmo3ParserEventContent struct {
	content string
}

type olmo3ParserEventToolCalls struct {
	calls []api.ToolCall
}

func (olmo3ParserEventContent) isOlmo3ParserEvent()   {}
func (olmo3ParserEventToolCalls) isOlmo3ParserEvent() {}

func (p *Olmo3Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	if done {
		// Drain any remaining content
		bufStr := p.buffer.String()
		p.buffer.Reset()
		if p.state == olmo3StateContent && len(bufStr) > 0 {
			return bufStr, "", nil, nil
		}
		return "", "", nil, nil
	}

	events := p.parseEvents()

	var contentSb strings.Builder
	var allCalls []api.ToolCall
	for _, event := range events {
		switch event := event.(type) {
		case olmo3ParserEventContent:
			contentSb.WriteString(event.content)
		case olmo3ParserEventToolCalls:
			allCalls = append(allCalls, event.calls...)
		}
	}

	return contentSb.String(), "", allCalls, nil
}

func (p *Olmo3Parser) parseEvents() []olmo3ParserEvent {
	var all []olmo3ParserEvent

	keepLooping := true
	for keepLooping {
		var events []olmo3ParserEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "olmo3 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Olmo3Parser) eat() ([]olmo3ParserEvent, bool) {
	var events []olmo3ParserEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case olmo3StateContent:
		if strings.Contains(bufStr, olmo3FuncCallsOpenTag) {
			// Found <function_calls> tag
			split := strings.SplitN(bufStr, olmo3FuncCallsOpenTag, 2)
			content := split[0]
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = olmo3StateToolCalls

			if len(content) > 0 {
				events = append(events, olmo3ParserEventContent{content: content})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, olmo3FuncCallsOpenTag); overlapLen > 0 {
			// Partial <function_calls> tag - withhold ambiguous content
			unambiguous := bufStr[:len(bufStr)-overlapLen]
			ambiguous := bufStr[len(bufStr)-overlapLen:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, olmo3ParserEventContent{content: unambiguous})
			}
			return events, false
		} else {
			// Regular content - emit all
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, olmo3ParserEventContent{content: bufStr})
			}
			return events, false
		}

	case olmo3StateToolCalls:
		if strings.Contains(bufStr, olmo3FuncCallsCloseTag) {
			// Found </function_calls> tag
			split := strings.SplitN(bufStr, olmo3FuncCallsCloseTag, 2)
			toolCallsStr := split[0]
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = olmo3StateToolCallsDone

			// Parse the function calls
			calls, err := parseOlmo3FunctionCalls(toolCallsStr)
			if err != nil {
				slog.Log(context.TODO(), logutil.LevelTrace, "failed to parse olmo3 function calls", "error", err, "content", toolCallsStr)
			} else if len(calls) > 0 {
				events = append(events, olmo3ParserEventToolCalls{calls: calls})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, olmo3FuncCallsCloseTag); overlapLen > 0 {
			// Partial </function_calls> tag - wait for more
			return events, false
		}
		// Still collecting tool calls, wait for close tag
		return events, false

	case olmo3StateToolCallsDone:
		// After tool calls, emit remaining content
		p.buffer.Reset()
		p.state = olmo3StateContent
		if len(bufStr) > 0 {
			events = append(events, olmo3ParserEventContent{content: bufStr})
		}
		return events, false
	}

	return events, false
}
