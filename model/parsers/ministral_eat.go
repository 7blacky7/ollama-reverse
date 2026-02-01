// Package parsers - Ministral Parser State-Machine
//
// Diese Datei enthält die eat()-Methode des MinistralParsers,
// die den Puffer konsumiert und Events basierend auf dem aktuellen
// Zustand generiert.
//
// Zustände:
// - ministralCollectingContent: Normaler Content
// - ministralCollectingThinkingContent: Innerhalb von [THINK]...[/THINK]
// - ministralCollectingToolName: Nach [TOOL_CALLS], vor [ARGS]
// - ministralCollectingToolArgs: Nach [ARGS], JSON sammeln
package parsers

import (
	"strings"
	"unicode"
)

// eat konsumiert den Parser-Puffer und gibt eindeutige Events zurück.
// Der zweite Rückgabewert zeigt an ob weiter geloopt werden soll
// (true bei Zustandswechsel, false wenn auf mehr Daten gewartet wird).
func (p *MinistralParser) eat() ([]ministralEvent, bool) {
	switch p.state {
	case ministralCollectingContent:
		return p.eatContent()

	case ministralCollectingThinkingContent:
		return p.eatThinking()

	case ministralCollectingToolName:
		return p.eatToolName()

	case ministralCollectingToolArgs:
		return p.eatToolArgs()

	default:
		panic("unexpected ministral event")
	}
}

// eatContent verarbeitet normalen Content
func (p *MinistralParser) eatContent() ([]ministralEvent, bool) {
	var events []ministralEvent
	bufStr := p.buffer.String()

	// Auf [TOOL_CALLS] Tag prüfen
	if strings.Contains(bufStr, ministralToolCallsTag) {
		split := strings.SplitN(bufStr, ministralToolCallsTag, 2)
		before := strings.TrimRightFunc(split[0], unicode.IsSpace)
		if len(before) > 0 {
			events = append(events, ministralEventContent{content: before})
		}
		after := split[1]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = ministralCollectingToolName
		return events, true
	}

	// Auf [THINK] Tag prüfen
	if strings.Contains(bufStr, ministralThinkTag) {
		split := strings.SplitN(bufStr, ministralThinkTag, 2)
		before := strings.TrimRightFunc(split[0], unicode.IsSpace)
		if len(before) > 0 {
			events = append(events, ministralEventContent{content: before})
		}
		after := split[1]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = ministralCollectingThinkingContent
		return events, true
	}

	// Auf partielle Tag-Überlappung prüfen
	overlapToolCalls := overlap(bufStr, ministralToolCallsTag)
	overlapThink := overlap(bufStr, ministralThinkTag)
	maxOverlap := max(overlapToolCalls, overlapThink)

	if maxOverlap > 0 {
		// Potentiellen partiellen Tag zurückhalten
		beforePartialTag := bufStr[:len(bufStr)-maxOverlap]
		trailingWS := trailingWhitespaceLen(beforePartialTag)
		ambiguousStart := len(beforePartialTag) - trailingWS
		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, ministralEventContent{content: unambiguous})
		}
		return events, false
	}

	// Kein Tag gefunden: Content ausgeben aber trailing Whitespace zurückhalten
	whitespaceLen := trailingWhitespaceLen(bufStr)
	ambiguousStart := len(bufStr) - whitespaceLen
	unambiguous := bufStr[:ambiguousStart]
	ambiguous := bufStr[ambiguousStart:]
	p.buffer.Reset()
	p.buffer.WriteString(ambiguous)
	if len(unambiguous) > 0 {
		events = append(events, ministralEventContent{content: unambiguous})
	}
	return events, false
}

// eatThinking verarbeitet Thinking-Content innerhalb von [THINK]...[/THINK]
func (p *MinistralParser) eatThinking() ([]ministralEvent, bool) {
	var events []ministralEvent
	bufStr := p.buffer.String()

	// Auf [/THINK] Tag prüfen
	if strings.Contains(bufStr, ministralThinkEndTag) {
		split := strings.SplitN(bufStr, ministralThinkEndTag, 2)
		thinkingContent := split[0]
		after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		if len(thinkingContent) > 0 {
			events = append(events, ministralEventThinking{thinking: thinkingContent})
		}
		p.state = ministralCollectingContent
		return events, true
	}

	// Auf partielle Überlappung mit [/THINK] prüfen
	if overlapLen := overlap(bufStr, ministralThinkEndTag); overlapLen > 0 {
		unambiguous := bufStr[:len(bufStr)-overlapLen]
		ambiguous := bufStr[len(bufStr)-overlapLen:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, ministralEventThinking{thinking: unambiguous})
		}
		return events, false
	}

	// Kein Tag gefunden: Allen Thinking-Content ausgeben
	p.buffer.Reset()
	if len(bufStr) > 0 {
		events = append(events, ministralEventThinking{thinking: bufStr})
	}
	return events, false
}

// eatToolName sammelt den Tool-Namen nach [TOOL_CALLS]
func (p *MinistralParser) eatToolName() ([]ministralEvent, bool) {
	var events []ministralEvent
	bufStr := p.buffer.String()

	// Auf [ARGS] Tag prüfen
	if strings.Contains(bufStr, ministralArgsTag) {
		split := strings.SplitN(bufStr, ministralArgsTag, 2)
		toolName := split[0]
		after := split[1]
		p.pendingToolName = toolName
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = ministralCollectingToolArgs
		return events, true
	}
	// Auf mehr Daten warten
	return events, false
}

// eatToolArgs sammelt die Tool-Argumente (JSON)
func (p *MinistralParser) eatToolArgs() ([]ministralEvent, bool) {
	var events []ministralEvent
	bufStr := p.buffer.String()
	jsonEnd := findJSONEnd(bufStr)

	if jsonEnd != -1 {
		jsonStr := bufStr[:jsonEnd+1]
		remaining := bufStr[jsonEnd+1:]

		events = append(events, ministralEventToolCall{
			name: p.pendingToolName,
			args: jsonStr,
		})

		p.pendingToolName = ""
		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = ministralCollectingContent
		return events, true
	}
	// Auf mehr Daten warten
	return events, false
}
