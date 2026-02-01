// interactive_main.go - Hauptloop fuer den interaktiven Modus
// Verarbeitet Benutzereingaben und koordiniert die Kommando-Verarbeitung
package cmd

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/readline"
)

// MultilineState repraesentiert den aktuellen Mehrzeilen-Eingabe-Status
type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
)

// generateInteractive startet den interaktiven Chat-Modus
func generateInteractive(cmd *cobra.Command, opts runOptions) error {
	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: "Press Enter to send",
	})
	if err != nil {
		return err
	}

	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	var sb strings.Builder
	var multiline MultilineState
	var thinkExplicitlySet bool = opts.Think != nil

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			sb.Reset()

			continue
		case err != nil:
			return err
		}

		switch {
		case multiline != MultilineNone:
			// check if there's a multiline terminating string
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				scanner.Prompt.UseAlt = true
				continue
			}

			switch multiline {
			case MultilineSystem:
				opts.System = sb.String()
				opts.Messages = append(opts.Messages, api.Message{Role: "system", Content: opts.System})
				fmt.Println("Set system message.")
				sb.Reset()
			}

			multiline = MultilineNone
			scanner.Prompt.UseAlt = false
		case strings.HasPrefix(line, `"""`):
			line := strings.TrimPrefix(line, `"""`)
			line, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(line)
			if !ok {
				// no multiline terminating string; need more input
				fmt.Fprintln(&sb)
				multiline = MultilinePrompt
				scanner.Prompt.UseAlt = true
			}
		case scanner.Pasting:
			fmt.Fprintln(&sb, line)
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/load"):
			if err := handleLoadCommand(cmd, &opts, line, &thinkExplicitlySet); err != nil {
				return err
			}
			continue
		case strings.HasPrefix(line, "/save"):
			if err := handleSaveCommand(cmd, &opts, line); err != nil {
				return err
			}
			continue
		case strings.HasPrefix(line, "/clear"):
			handleClearCommand(&opts)
			continue
		case strings.HasPrefix(line, "/set"):
			result := handleSetCommand(cmd, &opts, line, scanner, &sb, &thinkExplicitlySet)
			if result == commandContinue {
				continue
			} else if result == commandMultiline {
				multiline = MultilineSystem
				scanner.Prompt.UseAlt = true
				continue
			}
		case strings.HasPrefix(line, "/show"):
			if err := handleShowCommand(cmd, &opts, line); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			handleHelpCommand(line, opts)
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			isFile := false

			if opts.MultiModal {
				for _, f := range extractFileNames(line) {
					if strings.HasPrefix(f, args[0]) {
						isFile = true
						break
					}
				}
			}

			if !isFile {
				fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
				continue
			}

			sb.WriteString(line)
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == MultilineNone {
			newMessage := api.Message{Role: "user", Content: sb.String()}

			if opts.MultiModal {
				msg, images, err := extractFileData(sb.String())
				if err != nil {
					return err
				}

				newMessage.Content = msg
				newMessage.Images = images
			}

			opts.Messages = append(opts.Messages, newMessage)

			assistant, err := chat(cmd, opts)
			if err != nil {
				if strings.Contains(err.Error(), "does not support thinking") ||
					strings.Contains(err.Error(), "invalid think value") {
					fmt.Printf("error: %v\n", err)
					sb.Reset()
					continue
				}
				return err
			}
			if assistant != nil {
				opts.Messages = append(opts.Messages, *assistant)
			}

			sb.Reset()
		}
	}
}
