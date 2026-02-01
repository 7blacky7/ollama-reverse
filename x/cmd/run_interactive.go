// run_interactive.go - Interaktive Agent-Session (REPL)
//
// Dieses Modul enthält:
// - GenerateInteractive: Haupt-REPL-Loop für interaktive Agent-Sessions
// - initToolRegistry: Tool-Registry initialisieren
// - processCommand: Slash-Commands verarbeiten

package cmd

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/x/agent"
	"github.com/ollama/ollama/x/tools"
)

// GenerateInteractive führt eine interaktive Agent-Session aus.
// Wird von cmd.go aufgerufen wenn --experimental Flag gesetzt ist.
// Wenn yoloMode true ist, werden alle Tool-Approvals übersprungen.
// Wenn enableWebsearch true ist, wird das Web-Search-Tool registriert.
func GenerateInteractive(cmd *cobra.Command, modelName string, wordWrap bool, options map[string]any,
	think *api.ThinkValue, hideThinking bool, keepAlive *api.Duration, yoloMode bool, enableWebsearch bool) error {

	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: "Press Enter to send",
	})
	if err != nil {
		return err
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	// Tool-Registry und Capabilities initialisieren
	toolRegistry, supportsTools := initToolRegistry(cmd, modelName, enableWebsearch, yoloMode)

	// Approval-Manager für Session erstellen
	approval := agent.NewApprovalManager()

	var messages []api.Message
	var sb strings.Builder
	var format string
	var system string

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

		// Slash-Commands verarbeiten
		if strings.HasPrefix(line, "/") {
			shouldExit := processCommand(cmd, line, &modelName, &messages, &wordWrap, &think,
				&format, &system, options, scanner, approval, toolRegistry, supportsTools)
			if shouldExit {
				return nil
			}
			continue
		}

		sb.WriteString(line)

		if sb.Len() > 0 {
			newMessage := api.Message{Role: "user", Content: sb.String()}
			messages = append(messages, newMessage)

			verbose, _ := cmd.Flags().GetBool("verbose")
			opts := RunOptions{
				Model:        modelName,
				Messages:     messages,
				WordWrap:     wordWrap,
				Format:       format,
				Options:      options,
				Think:        think,
				HideThinking: hideThinking,
				KeepAlive:    keepAlive,
				Tools:        toolRegistry,
				Approval:     approval,
				YoloMode:     yoloMode,
				Verbose:      verbose,
			}

			assistant, err := Chat(cmd.Context(), opts)
			if err != nil {
				return err
			}
			if assistant != nil {
				messages = append(messages, *assistant)
			}

			sb.Reset()
		}
	}
}

// initToolRegistry initialisiert die Tool-Registry basierend auf Modell-Capabilities.
func initToolRegistry(cmd *cobra.Command, modelName string, enableWebsearch, yoloMode bool) (*tools.Registry, bool) {
	// Prüfen ob Modell Tools unterstützt
	supportsTools, err := checkModelCapabilities(cmd.Context(), modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not check model capabilities: %v\n", err)
		supportsTools = false
	}

	// Tool-Registry nur erstellen wenn Modell Tools unterstützt
	if !supportsTools {
		return nil, false
	}

	toolRegistry := tools.DefaultRegistry()

	// Web-Search und Web-Fetch Tools registrieren wenn via Flag aktiviert
	if enableWebsearch {
		toolRegistry.RegisterWebSearch()
		toolRegistry.RegisterWebFetch()
	}

	if toolRegistry.Has("bash") {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "This experimental version of Ollama has the \033[1mbash\033[0m tool enabled.")
		fmt.Fprintln(os.Stderr, "Models can read files on your computer, or run commands (after you allow them).")
		fmt.Fprintln(os.Stderr)
	}

	if toolRegistry.Has("web_search") || toolRegistry.Has("web_fetch") {
		fmt.Fprintln(os.Stderr, "The \033[1mWeb Search\033[0m and \033[1mWeb Fetch\033[0m tools are enabled. Models can search and fetch web content via ollama.com.")
		fmt.Fprintln(os.Stderr)
	}

	if yoloMode {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m yolo mode - all tool approvals will be skipped\n")
	}

	return toolRegistry, true
}

// processCommand verarbeitet Slash-Commands.
// Gibt true zurück wenn die Session beendet werden soll.
func processCommand(cmd *cobra.Command, line string, modelName *string, messages *[]api.Message,
	wordWrap *bool, think **api.ThinkValue, format, system *string, options map[string]any,
	scanner *readline.Instance, approval *agent.ApprovalManager,
	toolRegistry *tools.Registry, supportsTools bool) bool {

	switch {
	case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
		return true
	case strings.HasPrefix(line, "/clear"):
		*messages = []api.Message{}
		approval.Reset()
		fmt.Println("Cleared session context and tool approvals")
	case strings.HasPrefix(line, "/tools"):
		showToolsStatus(toolRegistry, approval, supportsTools)
	case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
		handleHelpCommand()
	case strings.HasPrefix(line, "/set"):
		args := strings.Fields(line)
		handleSetCommand(cmd, args, *modelName, scanner, wordWrap, think, format, system, options, messages)
	case strings.HasPrefix(line, "/show"):
		args := strings.Fields(line)
		handleShowCommand(cmd, args, *modelName, options, *system)
	case strings.HasPrefix(line, "/load"):
		args := strings.Fields(line)
		*modelName = handleLoadCommand(cmd, args, *modelName, *think, messages, approval)
	case strings.HasPrefix(line, "/save"):
		args := strings.Fields(line)
		handleSaveCommand(cmd, args, *modelName, options, *messages)
	default:
		fmt.Printf("Unknown command '%s'. Type /? for help\n", strings.Fields(line)[0])
	}
	return false
}
