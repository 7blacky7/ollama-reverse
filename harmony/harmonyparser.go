// Package harmony implementiert den Harmony-Parser fuer Ollama.
//
// Modul: harmonyparser.go - HarmonyParser Kernfunktionen
// Enthaelt: HarmonyParser Struktur, Events, Parsing-Logik
package harmony

import (
	"log/slog"
	"strings"
	"unicode"
)

type harmonyParserState int

const (
	harmonyParserState_LookingForMessageStart harmonyParserState = iota
	harmonyParserState_ParsingHeader
	harmonyParserState_ParsingContent
)

func (s harmonyParserState) String() string {
	switch s {
	// we're looking for the message start tag
	case harmonyParserState_LookingForMessageStart:
		return "LookingForMessageStart"
	case harmonyParserState_ParsingHeader:
		return "ParsingHeader"
	case harmonyParserState_ParsingContent:
		return "ParsingContent"
	default:
		return "Unknown"
	}
}

type HarmonyParser struct {
	state           harmonyParserState
	MessageStartTag string
	MessageEndTag   string
	HeaderEndTag    string
	acc             strings.Builder
	lifetimeAcc     strings.Builder
}

type HarmonyEvent interface {
	isHarmonyEvent()
}

type HarmonyEventMessageStart struct{}

func (HarmonyEventMessageStart) isHarmonyEvent() {}

type HarmonyEventHeaderComplete struct {
	Header HarmonyHeader
}

func (HarmonyEventHeaderComplete) isHarmonyEvent() {}

type HarmonyEventContentEmitted struct {
	Content string
}

func (HarmonyEventContentEmitted) isHarmonyEvent() {}

type HarmonyEventMessageEnd struct{}

func (HarmonyEventMessageEnd) isHarmonyEvent() {}

type HarmonyHeader struct {
	Role      string
	Channel   string
	Recipient string
}

func (s *HarmonyParser) AddImplicitStart() {
	s.acc.WriteString("<|start|>assistant")
}

func (s *HarmonyParser) AddImplicitStartOrPrefill(lastMessage *Message) {
	if lastMessage != nil && lastMessage.Role == "assistant" {
		// handle prefilling conditions
		if lastMessage.Content != "" {
			s.acc.WriteString("<|start|>assistant<|channel|>final<|message|>")
			return
		} else if lastMessage.Thinking != "" {
			s.acc.WriteString("<|start|>assistant<|channel|>analysis<|message|>")
			return
		}
	}
	s.AddImplicitStart()
}

func (s *HarmonyParser) AddContent(content string) []HarmonyEvent {
	s.lifetimeAcc.WriteString(content)
	s.acc.WriteString(content)

	var events []HarmonyEvent

	keepLooping := true
	// we loop because we might pass through multiple parsing states in a single
	// call to addContent, and we want to make sure callers don't have to wait for
	// data that's already unambiguous
	for keepLooping {
		var newEvents []HarmonyEvent
		newEvents, keepLooping = eat(s)
		events = append(events, newEvents...)
	}

	return events
}

// the additional bool return is true iff we should continue eating
func eat(s *HarmonyParser) ([]HarmonyEvent, bool) {
	switch s.state {
	case harmonyParserState_LookingForMessageStart:
		// does the acc contain the message start tag?
		if strings.Contains(s.acc.String(), s.MessageStartTag) {
			// split the acc into the message start tag and the rest
			split := strings.SplitN(s.acc.String(), s.MessageStartTag, 2)
			before := split[0]
			if before != "" {
				slog.Warn("harmony parser: found message start tag in the middle of the content", "content", s.acc.String())
			}
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_ParsingHeader
			return []HarmonyEvent{HarmonyEventMessageStart{}}, true
		}

		// no match, so we keep accumulating
		return nil, false
	case harmonyParserState_ParsingHeader:
		if strings.Contains(s.acc.String(), s.HeaderEndTag) {
			split := strings.SplitN(s.acc.String(), s.HeaderEndTag, 2)
			header := split[0]
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_ParsingContent
			return []HarmonyEvent{HarmonyEventHeaderComplete{Header: s.parseHeader(header)}}, true
		}
		return nil, false
	case harmonyParserState_ParsingContent:
		if strings.Contains(s.acc.String(), s.MessageEndTag) {
			// if we already have the message end tag, we can emit the content up to it
			split := strings.SplitN(s.acc.String(), s.MessageEndTag, 2)
			content := split[0]
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_LookingForMessageStart
			events := []HarmonyEvent{}
			if content != "" {
				events = append(events, HarmonyEventContentEmitted{Content: content})
			}
			events = append(events, HarmonyEventMessageEnd{})
			return events, true
		} else if overlapLen := overlap(s.acc.String(), s.MessageEndTag); overlapLen > 0 {
			// if our suffix contains the start of the message end tag, we can emit
			// the content up to the start of the message end tag
			content := s.acc.String()[:len(s.acc.String())-overlapLen]
			remaining := s.acc.String()[len(s.acc.String())-overlapLen:]
			s.acc.Reset()
			s.acc.WriteString(remaining)
			// emit the content we know isn't part of the message end tag, and keep
			// accumulating to disambiguate the rest
			if content == "" {
				return nil, false
			}
			return []HarmonyEvent{HarmonyEventContentEmitted{Content: content}}, false
		} else {
			// no end tag, so it's still normal content that we can immediately emit
			content := s.acc.String()
			if content == "" {
				return nil, false
			}
			s.acc.Reset()
			return []HarmonyEvent{HarmonyEventContentEmitted{Content: content}}, false
		}
	}

	return nil, false
}

func (s *HarmonyParser) parseHeader(raw string) HarmonyHeader {
	harmonyHeader := HarmonyHeader{}

	// if `<|constrain|>` is present, ensure it has a space before it so it gets
	// parsed as a separate token, even if the model didn't include the space
	if strings.Contains(raw, "<|constrain|>") {
		raw = strings.Replace(raw, "<|constrain|>", " <|constrain|>", 1)
		raw = strings.TrimSpace(raw)
	}

	// look for the optional channel tag, which is `<|channel|>` followed by the
	// channel name, all without any whitespace
	channelIndex := strings.Index(raw, "<|channel|>")
	if channelIndex != -1 {
		before := raw[:channelIndex]
		after := raw[channelIndex+len("<|channel|>"):]
		// the channel name is `after` all the way up to the first (if any) whitespace character
		idx := strings.IndexFunc(after, func(r rune) bool {
			return unicode.IsSpace(r)
		})
		if idx == -1 {
			idx = len(after)
		}
		harmonyHeader.Channel = after[:idx]
		after = after[idx:]
		// now we remove the channel tag from the raw string to further process
		raw = before + after
		raw = strings.TrimSpace(raw)
	}

	// split the header into whitespace-separated tokens
	tokens := strings.Fields(raw)

	// the first token is treated as the role
	if len(tokens) == 0 {
		slog.Error("harmony parser: missing role in header", "header", raw)
		return harmonyHeader
	}
	role := tokens[0]
	tokens = tokens[1:]
	// special case: if role starts with to= then it's a tool call
	if strings.HasPrefix(role, "to=") {
		harmonyHeader.Recipient = role[3:]
		harmonyHeader.Role = "tool"
	} else {
		harmonyHeader.Role = role
	}

	// the recipient (if any) can be specified before or after the channel tag, so
	// we check it at the end once we've already parsed the channel and role
	if harmonyHeader.Recipient == "" && len(tokens) > 0 && strings.HasPrefix(tokens[0], "to=") {
		harmonyHeader.Recipient = tokens[0][3:]
	}

	return harmonyHeader
}

// longest overlap between suffix of s and prefix of delim
func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

// Message ist ein lokaler Typ fuer die Parser-Kompatibilitaet
// Wird von api.Message abgeleitet
type Message struct {
	Role     string
	Content  string
	Thinking string
}
