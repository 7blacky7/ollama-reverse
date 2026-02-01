package parsers

import (
	"context"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type qwenParserState int

const (
	toolOpenTag  = "<tool_call>"
	toolCloseTag = "</tool_call>"
)

const (
	qwenParserState_LookingForToolStart qwenParserState = iota
	qwenParserState_CollectingToolContent
)

type Qwen3CoderParser struct {
	state qwenParserState
	acc   strings.Builder
	tools []api.Tool
}

func (p *Qwen3CoderParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3CoderParser) HasThinkingSupport() bool {
	return false
}

func (p *Qwen3CoderParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	return tools // Qwen doesn't modify tools
}

func (p *Qwen3CoderParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.acc.WriteString(s)

	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var sb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwenEventRawToolCall:
			toolCall, err := parseToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case qwenEventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here. See the note below about
			// `qwenEvent`s for more details
			sb.WriteString(event.content)
		}
	}

	return sb.String(), "", toolCalls, nil
}

func (p *Qwen3CoderParser) parseEvents() []qwenEvent {
	var all []qwenEvent

	keepLooping := true
	for keepLooping {
		var events []qwenEvent
		events, keepLooping = eat(p)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen events parsed", "events", all, "state", p.state, "acc", p.acc.String())
	}

	return all
}

// we use some internal event types in order to communicate between `Add` and
// `eat`. We do this to support interleaving content and parallel tool calls in
// the parser, even though qwen3-coder isn't supposed to do this. Our API
// doesn't currently support models outputting multiple messages in a turn, so
// we wouldn't be able to represent it yet, but there's no reason to prevent the
// parser from supporting it, especially for future models if they end up using
// a similar format.
type qwenEvent interface {
	isQwenEvent()
}

type qwenEventRawToolCall struct {
	raw string
}

type qwenEventContent struct {
	content string
}

func (qwenEventContent) isQwenEvent()     {}
func (qwenEventRawToolCall) isQwenEvent() {}

// eat consumes the parser's buffer, and returns a list of any unambiguous
// events from the current parser state. If the parser transitions to another
// state, it may have additional events to emit on the next call, which is what
// the second return value indicates
func eat(p *Qwen3CoderParser) ([]qwenEvent, bool) {
	var events []qwenEvent

	switch p.state {
	case qwenParserState_LookingForToolStart:
		if strings.Contains(p.acc.String(), toolOpenTag) {
			// we found a full tool open tag, so we can emit the content before the
			// tag, being sure to trim any trailing whitespace
			split := strings.SplitN(p.acc.String(), toolOpenTag, 2)
			before := split[0]
			before = strings.TrimRightFunc(before, unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			after := split[1]
			p.acc.Reset()
			p.acc.WriteString(after)
			p.state = qwenParserState_CollectingToolContent
			return events, true
		} else if overlap := overlap(p.acc.String(), toolOpenTag); overlap > 0 {
			// we found a partial tool open tag, so we can emit the unambiguous part,
			// which is the (trailing-whitespace trimmed) content before the partial
			// tool open tag
			beforePartialTag := p.acc.String()[:len(p.acc.String())-overlap]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen
			unambiguous := p.acc.String()[:ambiguousStart]
			ambiguous := p.acc.String()[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		} else {
			// we found content that is entirely not a tool call. We should withhold
			// any trailing whitespace in case this is the end of the content
			whitespaceLen := trailingWhitespaceLen(p.acc.String())
			ambiguousStart := len(p.acc.String()) - whitespaceLen
			unambiguous := p.acc.String()[:ambiguousStart]
			ambiguous := p.acc.String()[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		}
	case qwenParserState_CollectingToolContent:
		if strings.Contains(p.acc.String(), toolCloseTag) {
			split := strings.SplitN(p.acc.String(), toolCloseTag, 2)
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}
			// remove any whitespace between the tool call and any content after it
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.acc.Reset()
			p.acc.WriteString(after)
			events = append(events, qwenEventRawToolCall{raw: before})
			p.state = qwenParserState_LookingForToolStart
			return events, true
		} else {
			// note that we don't need to check the overlap here because we only plan
			// on parsing the tool call once we see the full closing tag. We don't
			// stream back the unparsed tool content, so there's no need to be eager
			// here
			return events, false
		}
	default:
		panic("unreachable")
	}
}
