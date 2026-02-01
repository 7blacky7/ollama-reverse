package parsers

import (
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type LFM2ParserState int

const (
	LFM2CollectingThinking LFM2ParserState = iota
	LFM2CollectingContent
	LFM2CollectingToolCalls
)

const (
	lfm2ThinkingOpenTag  = "<think>"
	lfm2ThinkingCloseTag = "</think>"
	lfm2ToolCallStartTag = "<|tool_call_start|>"
	lfm2ToolCallEndTag   = "<|tool_call_end|>"
)

type LFM2Parser struct {
	state                    LFM2ParserState
	buffer                   strings.Builder
	hasThinkingSupport       bool
	needsThinkingLeadingTrim bool // trim leading whitespace after <think> tag
	needsContentLeadingTrim  bool // trim leading whitespace after </think> tag
}

func (p *LFM2Parser) HasToolSupport() bool {
	return true
}

func (p *LFM2Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *LFM2Parser) setInitialState(lastMessage *api.Message, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Check both model capability AND request preference
	thinkingEnabled := p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !thinkingEnabled {
		p.state = LFM2CollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = LFM2CollectingContent
		return
	}

	p.state = LFM2CollectingThinking
	p.needsThinkingLeadingTrim = true
}

func (p *LFM2Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.setInitialState(lastMessage, thinkValue)
	return tools
}

type lfm2Event interface {
	isLFM2Event()
}

type lfm2EventThinkingContent struct {
	content string
}

type lfm2EventContent struct {
	content string
}

type lfm2EventToolCall struct {
	toolCall api.ToolCall
}

func (lfm2EventThinkingContent) isLFM2Event() {}
func (lfm2EventContent) isLFM2Event()         {}
func (lfm2EventToolCall) isLFM2Event()        {}

func (p *LFM2Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case lfm2EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case lfm2EventThinkingContent:
			thinkingSb.WriteString(event.content)
		case lfm2EventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *LFM2Parser) parseEvents() []lfm2Event {
	var all []lfm2Event

	keepLooping := true
	for keepLooping {
		var events []lfm2Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *LFM2Parser) eat() ([]lfm2Event, bool) {
	var events []lfm2Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case LFM2CollectingThinking:
		// Strip opening <think> tag if present
		if strings.HasPrefix(bufStr, lfm2ThinkingOpenTag) {
			bufStr = bufStr[len(lfm2ThinkingOpenTag):]
			p.needsThinkingLeadingTrim = true
			p.buffer.Reset()
			p.buffer.WriteString(bufStr)
		}

		// Trim leading whitespace after <think> tag (may span multiple chunks)
		if p.needsThinkingLeadingTrim {
			if trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace); trimmed != bufStr {
				bufStr = trimmed
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
			}
			// Clear flag once we have non-whitespace content or buffer is empty
			if len(bufStr) > 0 {
				p.needsThinkingLeadingTrim = false
			}
		}

		if strings.Contains(bufStr, lfm2ThinkingCloseTag) { // thinking[</think>] -> content
			split := strings.SplitN(bufStr, lfm2ThinkingCloseTag, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = LFM2CollectingContent
			p.needsThinkingLeadingTrim = false
			// Set flag to trim any additional whitespace that may arrive in later chunks
			p.needsContentLeadingTrim = len(remaining) == 0

			if len(thinking) > 0 {
				events = append(events, lfm2EventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, lfm2ThinkingCloseTag); overlapLen > 0 { // partial </think>
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, lfm2EventThinkingContent{content: unambiguous})
			}
			return events, false
		} else { // otherwise its thinking content
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, lfm2EventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case LFM2CollectingContent:
		// Trim leading whitespace after </think> tag (may span multiple chunks)
		if p.needsContentLeadingTrim {
			if trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace); trimmed != bufStr {
				bufStr = trimmed
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
			}
			// Clear flag once we have non-whitespace content
			if len(bufStr) > 0 {
				p.needsContentLeadingTrim = false
			}
		}

		if strings.Contains(bufStr, lfm2ToolCallStartTag) { // content[<|tool_call_start|>] -> tool calls
			split := strings.SplitN(bufStr, lfm2ToolCallStartTag, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = LFM2CollectingToolCalls

			if len(contentBefore) > 0 {
				events = append(events, lfm2EventContent{content: contentBefore})
			}
			return events, true
		} else { // otherwise its content
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, lfm2EventContent{content: bufStr})
			}
			return events, false
		}

	case LFM2CollectingToolCalls:
		// Look for complete tool call JSON between tags
		if idx := strings.Index(bufStr, lfm2ToolCallEndTag); idx != -1 {
			toolCallContent := bufStr[:idx]

			if toolCalls, err := p.parseToolCallsContent(toolCallContent); err == nil && len(toolCalls) > 0 {
				remaining := bufStr[idx+len(lfm2ToolCallEndTag):]

				// Check if there's another tool call
				if strings.HasPrefix(remaining, lfm2ToolCallStartTag) {
					remaining = remaining[len(lfm2ToolCallStartTag):]
				} else {
					// No more tool calls, go back to content
					remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
					p.state = LFM2CollectingContent
				}

				p.buffer.Reset()
				p.buffer.WriteString(remaining)

				for _, tc := range toolCalls {
					events = append(events, lfm2EventToolCall{toolCall: tc})
				}
				return events, true
			} else if err != nil {
				slog.Warn("lfm2 tool call parsing failed", "error", err, "content", toolCallContent)
			}
		}

		return events, false
	}

	return events, false
}
