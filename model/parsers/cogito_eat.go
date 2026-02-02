// =============================================================================
// Modul: cogito_eat.go
// Beschreibung: State-Machine-Implementierung für den Cogito-Parser.
//               Enthält die eat()-Funktion mit allen State-Handlern für
//               Thinking, Content, ToolCalls und ToolOutput.
// =============================================================================

package parsers

import (
	"log/slog"
	"strings"
	"unicode"
)

// eat verarbeitet den Buffer basierend auf dem aktuellen Parser-State.
// Gibt Events und einen Boolean zurück, der anzeigt ob weiter geloopt werden soll.
func (p *CogitoParser) eat() ([]cogitoEvent, bool) {
	var events []cogitoEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case CogitoCollectingThinking:
		return p.eatThinking(bufStr)
	case CogitoCollectingContent:
		return p.eatContent(bufStr)
	case CogitoCollectingToolCalls:
		return p.eatToolCalls(bufStr)
	case CogitoCollectingToolOutput:
		return p.eatToolOutput(bufStr)
	}

	return events, false
}

// eatThinking verarbeitet den Thinking-State.
// Sucht nach dem </think>-Tag und extrahiert Thinking-Content.
func (p *CogitoParser) eatThinking(bufStr string) ([]cogitoEvent, bool) {
	var events []cogitoEvent

	// Vollständiger </think>-Tag gefunden -> Wechsel zu Content
	if strings.Contains(bufStr, cogitoThinkingCloseTag) {
		split := strings.SplitN(bufStr, cogitoThinkingCloseTag, 2)
		thinking := split[0]
		thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

		remaining := split[1]
		remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = CogitoCollectingContent

		if len(thinking) > 0 {
			events = append(events, cogitoEventThinkingContent{content: thinking})
		}
		return events, true
	}

	// Partieller </think>-Tag am Ende -> Ambigue Region behalten
	if overlapLen := overlap(bufStr, cogitoThinkingCloseTag); overlapLen > 0 {
		beforePartialTag := bufStr[:len(bufStr)-overlapLen]
		trailingLen := trailingWhitespaceLen(beforePartialTag)
		ambiguousStart := len(beforePartialTag) - trailingLen

		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, cogitoEventThinkingContent{content: unambiguous})
		}
		return events, false
	}

	// Kein Tag gefunden -> Thinking Content ausgeben (Whitespace am Ende behalten)
	whitespaceLen := trailingWhitespaceLen(bufStr)
	ambiguousStart := len(bufStr) - whitespaceLen

	unambiguous := bufStr[:ambiguousStart]
	ambiguous := bufStr[ambiguousStart:]
	p.buffer.Reset()
	p.buffer.WriteString(ambiguous)
	if len(unambiguous) > 0 {
		events = append(events, cogitoEventThinkingContent{content: unambiguous})
	}
	return events, false
}

// eatContent verarbeitet den Content-State.
// Sucht nach Tool-Call- oder Tool-Output-Begin-Tags.
func (p *CogitoParser) eatContent(bufStr string) ([]cogitoEvent, bool) {
	var events []cogitoEvent

	switch {
	// Tool-Calls beginnen -> Wechsel zu ToolCalls-State
	case strings.Contains(bufStr, cogitoToolCallsBeginTag):
		split := strings.SplitN(bufStr, cogitoToolCallsBeginTag, 2)
		contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
		remaining := split[1]

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = CogitoCollectingToolCalls

		if len(contentBefore) > 0 {
			events = append(events, cogitoEventContent{content: contentBefore})
		}
		return events, true

	// Tool-Outputs beginnen -> Wechsel zu ToolOutput-State
	case strings.Contains(bufStr, cogitoToolOutputsBeginTag):
		split := strings.SplitN(bufStr, cogitoToolOutputsBeginTag, 2)
		contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
		remaining := split[1]

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = CogitoCollectingToolOutput

		if len(contentBefore) > 0 {
			events = append(events, cogitoEventContent{content: contentBefore})
		}
		return events, true

	// Normaler Content
	default:
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, cogitoEventContent{content: bufStr})
		}
		return events, false
	}
}

// eatToolCalls verarbeitet den ToolCalls-State.
// Sucht nach einzelnen Tool-Call-Begin/End-Tags und dem ToolCalls-End-Tag.
func (p *CogitoParser) eatToolCalls(bufStr string) ([]cogitoEvent, bool) {
	var events []cogitoEvent

	// Einzelnen Tool-Call suchen und parsen
	if idx := strings.Index(bufStr, cogitoToolCallBeginTag); idx != -1 {
		startIdx := idx + len(cogitoToolCallBeginTag)
		if endIdx := strings.Index(bufStr[startIdx:], cogitoToolCallEndTag); endIdx != -1 {
			toolCallContent := bufStr[startIdx : startIdx+endIdx]

			if toolCall, err := p.parseToolCallContent(toolCallContent); err == nil {
				remaining := bufStr[startIdx+endIdx+len(cogitoToolCallEndTag):]
				remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

				p.buffer.Reset()
				p.buffer.WriteString(remaining)

				events = append(events, cogitoEventToolCall{toolCall: toolCall})
				return events, true
			} else {
				slog.Warn("cogito tool call parsing failed", "error", err)
			}
		}
	}

	// Ende aller Tool-Calls -> zurück zu Content
	if idx := strings.Index(bufStr, cogitoToolCallsEndTag); idx != -1 {
		remaining := bufStr[idx+len(cogitoToolCallsEndTag):]
		remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = CogitoCollectingContent

		return events, true
	}

	return events, false
}

// eatToolOutput verarbeitet den ToolOutput-State.
// Sucht nach einzelnen Tool-Output-Begin/End-Tags und dem ToolOutputs-End-Tag.
func (p *CogitoParser) eatToolOutput(bufStr string) ([]cogitoEvent, bool) {
	var events []cogitoEvent

	// Einzelnen Tool-Output suchen (wird ignoriert, aber Buffer wird aufgeräumt)
	if idx := strings.Index(bufStr, cogitoToolOutputBeginTag); idx != -1 {
		startIdx := idx + len(cogitoToolOutputBeginTag)
		if endIdx := strings.Index(bufStr[startIdx:], cogitoToolOutputEndTag); endIdx != -1 {
			remaining := bufStr[startIdx+endIdx+len(cogitoToolOutputEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)

			return events, true
		}
	}

	// Ende aller Tool-Outputs -> zurück zu Content
	if idx := strings.Index(bufStr, cogitoToolOutputsEndTag); idx != -1 {
		remaining := bufStr[idx+len(cogitoToolOutputsEndTag):]
		remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = CogitoCollectingContent

		return events, true
	}

	return events, false
}
