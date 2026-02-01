// run_display.go - Anzeige-Logik für Chat-Output
//
// Dieses Modul enthält:
// - displayResponse: Haupt-Display-Funktion mit Word-Wrap
// - displayResponseState: State für Word-Wrap-Logik
// - thinkingOutputOpeningText/ClosingText: Thinking-Anzeige
// - renderToolCalls: Tool-Calls formatieren
// - truncateUTF8: UTF8-sichere String-Kürzung
// - formatToolShort: Kurze Tool-Beschreibung

package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/x/agent"
	"github.com/ollama/ollama/x/tools"
)

// displayResponseState hält den Zustand für Word-Wrap-Anzeige.
type displayResponseState struct {
	lineLength int
	wordBuffer string
}

// displayResponse zeigt Content mit optionalem Word-Wrap an.
func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if len(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				// Länge des letzten Worts zurückgehen und bis Zeilenende löschen
				a := len(state.wordBuffer)
				if a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Printf("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)

				state.lineLength = len(state.wordBuffer) + 1
			} else {
				fmt.Print(string(ch))
				state.lineLength++

				switch ch {
				case ' ', '\t':
					state.wordBuffer = ""
				case '\n', '\r':
					state.lineLength = 0
					state.wordBuffer = ""
				default:
					state.wordBuffer += string(ch)
				}
			}
		}
	} else {
		fmt.Printf("%s%s", state.wordBuffer, content)
		if len(state.wordBuffer) > 0 {
			state.wordBuffer = ""
		}
	}
}

// thinkingOutputOpeningText gibt den Öffnungstext für Thinking-Output zurück.
func thinkingOutputOpeningText(plainText bool) string {
	text := "Thinking...\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault + readline.ColorGrey
}

// thinkingOutputClosingText gibt den Schließungstext für Thinking-Output zurück.
func thinkingOutputClosingText(plainText bool) string {
	text := "...done thinking.\n\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault
}

// renderToolCalls formatiert Tool-Calls für die Anzeige.
func renderToolCalls(toolCalls []api.ToolCall, plainText bool) string {
	out := ""
	formatExplanation := ""
	formatValues := ""
	if !plainText {
		formatExplanation = readline.ColorGrey + readline.ColorBold
		formatValues = readline.ColorDefault
		out += formatExplanation
	}
	for i, toolCall := range toolCalls {
		argsAsJSON, err := json.Marshal(toolCall.Function.Arguments)
		if err != nil {
			return ""
		}
		if i > 0 {
			out += "\n"
		}
		out += fmt.Sprintf("  Tool call: %s(%s)", formatValues+toolCall.Function.Name+formatExplanation, formatValues+string(argsAsJSON)+formatExplanation)
	}
	if !plainText {
		out += readline.ColorDefault
	}
	return out
}

// truncateUTF8 kürzt einen String sicher auf maximal limit Runes, fügt "..." hinzu wenn gekürzt.
func truncateUTF8(s string, limit int) string {
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

// formatToolShort gibt eine kurze Beschreibung eines Tool-Calls zurück.
func formatToolShort(toolName string, args map[string]any) string {
	displayName := agent.ToolDisplayName(toolName)
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			return fmt.Sprintf("%s: %s", displayName, truncateUTF8(cmd, 50))
		}
	}
	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			return fmt.Sprintf("%s: %s", displayName, truncateUTF8(query, 50))
		}
	}
	return displayName
}

// showToolsStatus zeigt den aktuellen Tool- und Approval-Status an.
func showToolsStatus(registry *tools.Registry, approval *agent.ApprovalManager, supportsTools bool) {
	if !supportsTools || registry == nil {
		fmt.Println("Tools not available - model does not support tool calling")
		fmt.Println()
		return
	}

	fmt.Println("Available tools:")
	for _, name := range registry.Names() {
		tool, _ := registry.Get(name)
		fmt.Printf("  %s - %s\n", name, tool.Description())
	}

	allowed := approval.AllowedTools()
	if len(allowed) > 0 {
		fmt.Println("\nSession approvals:")
		for _, key := range allowed {
			fmt.Printf("  %s\n", key)
		}
	} else {
		fmt.Println("\nNo tools approved for this session yet")
	}
	fmt.Println()
}
