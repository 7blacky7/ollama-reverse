// Package agent provides agent loop orchestration and tool approval.
// Datei: approval_ui.go
// Inhalt: UI-Hilfsfunktionen fuer Tool-Approval (formatToolDisplay, wrapText, PromptYesNo)
package agent

import (
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"
)

// formatToolDisplay creates the display string for a tool call.
func formatToolDisplay(toolName string, args map[string]any) string {
	var sb strings.Builder
	displayName := ToolDisplayName(toolName)

	// For bash, show command directly
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("Command: %s", cmd))
			return sb.String()
		}
	}

	// For web search, show query and internet notice
	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("Query: %s\n", query))
			sb.WriteString("Uses internet via ollama.com")
			return sb.String()
		}
	}

	// For web fetch, show URL and internet notice
	if toolName == "web_fetch" {
		if url, ok := args["url"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("URL: %s\n", url))
			sb.WriteString("Uses internet via ollama.com")
			return sb.String()
		}
	}

	// Generic display
	sb.WriteString(fmt.Sprintf("Tool: %s", displayName))
	if len(args) > 0 {
		sb.WriteString("\nArguments: ")
		first := true
		for k, v := range args {
			if !first {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s=%v", k, v))
			first = false
		}
	}
	return sb.String()
}

// wrapText wraps text to fit within maxWidth, returning lines
func wrapText(text string, maxWidth int) []string {
	if maxWidth < 5 {
		maxWidth = 5
	}
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if len(line) <= maxWidth {
			lines = append(lines, line)
			continue
		}
		// Wrap long lines
		for len(line) > maxWidth {
			// Try to break at space
			breakAt := maxWidth
			for i := maxWidth; i > maxWidth/2; i-- {
				if i < len(line) && line[i] == ' ' {
					breakAt = i
					break
				}
			}
			lines = append(lines, line[:breakAt])
			line = strings.TrimLeft(line[breakAt:], " ")
		}
		if len(line) > 0 {
			lines = append(lines, line)
		}
	}
	return lines
}

// PromptYesNo displays a simple Yes/No prompt and returns the user's choice.
// Returns true for Yes, false for No.
func PromptYesNo(question string) (bool, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	selected := 0 // 0 = Yes, 1 = No
	options := []string{"Yes", "No"}

	// Hide cursor
	fmt.Fprint(os.Stderr, "\033[?25l")
	defer fmt.Fprint(os.Stderr, "\033[?25h")

	renderYesNo := func() {
		// Move to start of line and clear
		fmt.Fprintf(os.Stderr, "\r\033[K")
		fmt.Fprintf(os.Stderr, "%s  ", question)
		for i, opt := range options {
			if i == selected {
				fmt.Fprintf(os.Stderr, "\033[1m%s\033[0m  ", opt)
			} else {
				fmt.Fprintf(os.Stderr, "\033[37m%s\033[0m  ", opt)
			}
		}
	}

	renderYesNo()

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			return false, err
		}

		if n == 1 {
			switch buf[0] {
			case 'y', 'Y':
				selected = 0
				renderYesNo()
			case 'n', 'N':
				selected = 1
				renderYesNo()
			case '\r', '\n': // Enter
				fmt.Fprintf(os.Stderr, "\r\033[K") // Clear line
				return selected == 0, nil
			case 3: // Ctrl+C
				fmt.Fprintf(os.Stderr, "\r\033[K")
				return false, nil
			case 27: // Escape - could be arrow key
				// Read more bytes for arrow keys
				continue
			}
		} else if n == 3 && buf[0] == 27 && buf[1] == 91 {
			// Arrow keys
			switch buf[2] {
			case 'D': // Left
				if selected > 0 {
					selected--
				}
				renderYesNo()
			case 'C': // Right
				if selected < len(options)-1 {
					selected++
				}
				renderYesNo()
			}
		}
	}
}
