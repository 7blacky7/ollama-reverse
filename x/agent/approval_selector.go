// Package agent provides agent loop orchestration and tool approval.
// Datei: approval_selector.go
// Inhalt: Interaktiver Terminal-Selector fuer Tool-Approval (Rendering, Input-Verarbeitung)
package agent

import (
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"
)

// selectorState holds the state for the interactive selector
type selectorState struct {
	toolDisplay    string
	selected       int
	totalLines     int
	termWidth      int
	termHeight     int
	boxWidth       int
	innerWidth     int
	denyReason     string // deny reason (always visible in box)
	isWarning      bool   // true if command has warning
	warningMessage string // dynamic warning message to display
	allowlistInfo  string // show what will be allowlisted
}

// runSelector runs the interactive selector and returns the selected index and optional deny reason.
func runSelector(fd int, oldState *term.State, toolDisplay string, isWarning bool, warningMessage string, allowlistInfo string) (int, string, error) {
	state := &selectorState{
		toolDisplay: toolDisplay, selected: 0, isWarning: isWarning,
		warningMessage: warningMessage, allowlistInfo: allowlistInfo,
	}

	// Get terminal size
	state.termWidth, state.termHeight, _ = term.GetSize(fd)
	if state.termWidth < 20 {
		state.termWidth = 80
	}

	// Calculate box width: 90% of terminal, min 24, max 60
	state.boxWidth = (state.termWidth * 90) / 100
	if state.boxWidth > 60 {
		state.boxWidth = 60
	}
	if state.boxWidth < 24 {
		state.boxWidth = 24
	}
	if state.boxWidth > state.termWidth-1 {
		state.boxWidth = state.termWidth - 1
	}
	state.innerWidth = state.boxWidth - 4
	state.totalLines = calculateTotalLines(state)

	fmt.Fprint(os.Stderr, "\033[?25l")         // Hide cursor
	defer fmt.Fprint(os.Stderr, "\033[?25h") // Show cursor when done

	renderSelectorBox(state)
	numOptions := len(optionLabels)

	for {
		buf := make([]byte, 8)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			clearSelectorBox(state)
			return 2, "", err
		}

		for i := 0; i < n; i++ {
			ch := buf[i]
			// Arrow keys
			if ch == 27 && i+2 < n && buf[i+1] == '[' {
				oldSel := state.selected
				if buf[i+2] == 'A' && state.selected > 0 {
					state.selected--
				} else if buf[i+2] == 'B' && state.selected < numOptions-1 {
					state.selected++
				}
				if oldSel != state.selected {
					updateSelectorOptions(state)
				}
				i += 2
				continue
			}

			switch {
			case ch == 3: // Ctrl+C
				clearSelectorBox(state)
				return -1, "", nil
			case ch == 13: // Enter
				clearSelectorBox(state)
				if state.selected == 2 {
					return 2, state.denyReason, nil
				}
				return state.selected, "", nil
			case ch >= '1' && ch <= '3': // Quick select
				sel := int(ch - '1')
				clearSelectorBox(state)
				if sel == 2 {
					return 2, state.denyReason, nil
				}
				return sel, "", nil
			case ch == 127 || ch == 8: // Backspace
				if len(state.denyReason) > 0 {
					runes := []rune(state.denyReason)
					state.denyReason = string(runes[:len(runes)-1])
					updateReasonInput(state)
				}
			case ch == 27 && len(state.denyReason) > 0: // Escape
				state.denyReason = ""
				updateReasonInput(state)
			case ch >= 32 && ch < 127: // Printable
				maxLen := state.innerWidth - 2
				if maxLen < 10 {
					maxLen = 10
				}
				if len(state.denyReason) < maxLen {
					state.denyReason += string(ch)
					if state.selected != 2 {
						state.selected = 2
						updateSelectorOptions(state)
					} else {
						updateReasonInput(state)
					}
				}
			}
		}
	}
}

// getHintLines returns the hint text wrapped to terminal width
func getHintLines(state *selectorState) []string {
	hint := "up/down select, enter confirm, 1-3 quick select, ctrl+c cancel"
	if state.termWidth >= len(hint)+1 {
		return []string{hint}
	}
	return wrapText(hint, state.termWidth-1)
}

// calculateTotalLines calculates how many lines the selector will use
func calculateTotalLines(state *selectorState) int {
	toolLines := strings.Split(state.toolDisplay, "\n")
	hintLines := getHintLines(state)
	warningLines := 0
	if state.isWarning {
		warningLines = 2
	}
	return warningLines + len(toolLines) + 1 + len(optionLabels) + 1 + len(hintLines)
}

// renderOptionLine renders a single option line
func renderOptionLine(state *selectorState, i int, label string) {
	if i == 2 { // Deny option
		denyLabel := "3. Deny: "
		inputDisplay := state.denyReason
		if inputDisplay == "" {
			inputDisplay = "\033[90m(optional reason)\033[0m"
		}
		if i == state.selected {
			fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
		} else {
			fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
		}
	} else {
		displayLabel := label
		if i == 1 && state.allowlistInfo != "" {
			displayLabel = fmt.Sprintf("%s  \033[90m%s\033[0m", label, state.allowlistInfo)
		}
		if i == state.selected {
			fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m\033[K\r\n", displayLabel)
		} else {
			fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m\033[K\r\n", displayLabel)
		}
	}
}

// renderHintLines renders the hint lines at the bottom
func renderHintLines(hintLines []string) {
	fmt.Fprintf(os.Stderr, "\033[K\r\n")
	for i, line := range hintLines {
		if i == len(hintLines)-1 {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K", line)
		} else {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K\r\n", line)
		}
	}
}

// renderSelectorBox renders the selector (minimal, no box)
func renderSelectorBox(state *selectorState) {
	toolLines := strings.Split(state.toolDisplay, "\n")

	if state.isWarning {
		msg := state.warningMessage
		if msg == "" {
			msg = "command targets paths outside project"
		}
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m %s\033[K\r\n\033[K\r\n", msg)
	}

	for _, line := range toolLines {
		fmt.Fprintf(os.Stderr, "%s\033[K\r\n", line)
	}
	fmt.Fprintf(os.Stderr, "\033[K\r\n")

	for i, label := range optionLabels {
		renderOptionLine(state, i, label)
	}
	renderHintLines(getHintLines(state))
}

// updateSelectorOptions updates just the options portion of the selector
func updateSelectorOptions(state *selectorState) {
	hintLines := getHintLines(state)
	linesToMove := len(hintLines) - 1 + 1 + len(optionLabels)
	fmt.Fprintf(os.Stderr, "\033[%dA\r", linesToMove)

	for i, label := range optionLabels {
		renderOptionLine(state, i, label)
	}
	renderHintLines(hintLines)
}

// updateReasonInput updates just the Deny option line
func updateReasonInput(state *selectorState) {
	hintLines := getHintLines(state)
	linesToMove := len(hintLines) - 1 + 1 + 1
	fmt.Fprintf(os.Stderr, "\033[%dA\r", linesToMove)

	renderOptionLine(state, 2, optionLabels[2])
	renderHintLines(hintLines)
}

// clearSelectorBox clears the selector from screen
func clearSelectorBox(state *selectorState) {
	fmt.Fprint(os.Stderr, "\r\033[K")
	for range state.totalLines - 1 {
		fmt.Fprint(os.Stderr, "\033[A\033[K")
	}
	fmt.Fprint(os.Stderr, "\r")
}
