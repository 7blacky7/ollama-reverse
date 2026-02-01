// Modul: selector_render.go
// Beschreibung: Rendering-Funktionen für das Selector-UI.
// Zeichnet Single-Select und Multi-Select Listen im Terminal.

package config

import (
	"fmt"
	"io"
)

func renderSelect(w io.Writer, prompt string, s *selectState) int {
	filtered := s.filtered()

	if s.filter == "" {
		fmt.Fprintf(w, "%s %sType to filter...%s\r\n", prompt, ansiGray, ansiReset)
	} else {
		fmt.Fprintf(w, "%s %s\r\n", prompt, s.filter)
	}
	lineCount := 1

	if len(filtered) == 0 {
		fmt.Fprintf(w, "  %s(no matches)%s\r\n", ansiGray, ansiReset)
		lineCount++
	} else {
		displayCount := min(len(filtered), maxDisplayedItems)

		for i := range displayCount {
			idx := s.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			item := filtered[idx]
			prefix := "    "
			if idx == s.selected {
				prefix = "  " + ansiBold + "> "
			}
			if item.Description != "" {
				fmt.Fprintf(w, "%s%s%s %s- %s%s\r\n", prefix, item.Name, ansiReset, ansiGray, item.Description, ansiReset)
			} else {
				fmt.Fprintf(w, "%s%s%s\r\n", prefix, item.Name, ansiReset)
			}
			lineCount++
		}

		if remaining := len(filtered) - s.scrollOffset - displayCount; remaining > 0 {
			fmt.Fprintf(w, "  %s... and %d more%s\r\n", ansiGray, remaining, ansiReset)
			lineCount++
		}
	}

	return lineCount
}

func renderMultiSelect(w io.Writer, prompt string, s *multiSelectState) int {
	filtered := s.filtered()

	if s.filter == "" {
		fmt.Fprintf(w, "%s %sType to filter...%s\r\n", prompt, ansiGray, ansiReset)
	} else {
		fmt.Fprintf(w, "%s %s\r\n", prompt, s.filter)
	}
	lineCount := 1

	if len(filtered) == 0 {
		fmt.Fprintf(w, "  %s(no matches)%s\r\n", ansiGray, ansiReset)
		lineCount++
	} else {
		displayCount := min(len(filtered), maxDisplayedItems)

		for i := range displayCount {
			idx := s.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			item := filtered[idx]
			origIdx := s.itemIndex[item.Name]

			checkbox := "[ ]"
			if s.checked[origIdx] {
				checkbox = "[x]"
			}

			prefix := "  "
			suffix := ""
			if idx == s.highlighted && !s.focusOnButton {
				prefix = "> "
			}
			if len(s.checkOrder) > 0 && s.checkOrder[0] == origIdx {
				suffix = " " + ansiGray + "(default)" + ansiReset
			}

			if idx == s.highlighted && !s.focusOnButton {
				fmt.Fprintf(w, "  %s%s %s %s%s%s\r\n", ansiBold, prefix, checkbox, item.Name, ansiReset, suffix)
			} else {
				fmt.Fprintf(w, "  %s %s %s%s\r\n", prefix, checkbox, item.Name, suffix)
			}
			lineCount++
		}

		if remaining := len(filtered) - s.scrollOffset - displayCount; remaining > 0 {
			fmt.Fprintf(w, "  %s... and %d more%s\r\n", ansiGray, remaining, ansiReset)
			lineCount++
		}
	}

	fmt.Fprintf(w, "\r\n")
	lineCount++
	count := s.selectedCount()
	switch {
	case count == 0:
		fmt.Fprintf(w, "  %sSelect at least one model.%s\r\n", ansiGray, ansiReset)
	case s.focusOnButton:
		fmt.Fprintf(w, "  %s> [ Continue ]%s %s(%d selected)%s\r\n", ansiBold, ansiReset, ansiGray, count, ansiReset)
	default:
		fmt.Fprintf(w, "    %s[ Continue ] (%d selected) - press Tab%s\r\n", ansiGray, count, ansiReset)
	}
	lineCount++

	return lineCount
}
