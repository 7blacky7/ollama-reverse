// cmd_display.go - Display und Output-Funktionen
// Hauptfunktionen: displayResponse, renderToolCalls, thinking*
package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/mattn/go-runewidth"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/readline"
)

// displayResponse - Zeigt Antwort-Text mit optionalem Word-Wrap an
func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if runewidth.StringWidth(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				a := runewidth.StringWidth(state.wordBuffer)
				if a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Printf("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)
				chWidth := runewidth.RuneWidth(ch)

				state.lineLength = runewidth.StringWidth(state.wordBuffer) + chWidth
			} else {
				fmt.Print(string(ch))
				state.lineLength += runewidth.RuneWidth(ch)
				if runewidth.RuneWidth(ch) >= 2 {
					state.wordBuffer = ""
					continue
				}

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

// thinkingOutputOpeningText - Oeffnungstext fuer Thinking-Modus
func thinkingOutputOpeningText(plainText bool) string {
	text := "Thinking...\n"
	if plainText {
		return text
	}
	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault + readline.ColorGrey
}

// thinkingOutputClosingText - Schliessungstext fuer Thinking-Modus
func thinkingOutputClosingText(plainText bool) string {
	text := "...done thinking.\n\n"
	if plainText {
		return text
	}
	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault
}

// renderToolCalls - Rendert Tool-Calls fuer Ausgabe
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
		out += fmt.Sprintf("  Model called a non-existent function '%s()' with arguments: %s", formatValues+toolCall.Function.Name+formatExplanation, formatValues+string(argsAsJSON)+formatExplanation)
	}
	if !plainText {
		out += readline.ColorDefault
	}
	return out
}
