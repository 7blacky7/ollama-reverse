// run_commands.go - Interaktive Slash-Commands
//
// Dieses Modul enthält:
// - handleHelpCommand: /help Hilfe-Anzeige
// - handleSetCommand: /set Parameter-Kommando
// - handleLoadCommand: /load Modell-Wechsel
// - handleSaveCommand: /save Session-Speicherung
//
// Siehe auch: run_commands_show.go für /show Kommando

package cmd

import (
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/agent"
)

// handleHelpCommand zeigt die verfügbaren Befehle an.
func handleHelpCommand() {
	fmt.Fprintln(os.Stderr, "Available Commands:")
	fmt.Fprintln(os.Stderr, "  /set            Set session variables")
	fmt.Fprintln(os.Stderr, "  /show           Show model information")
	fmt.Fprintln(os.Stderr, "  /load           Load a different model")
	fmt.Fprintln(os.Stderr, "  /save           Save session as a model")
	fmt.Fprintln(os.Stderr, "  /tools          Show available tools and approvals")
	fmt.Fprintln(os.Stderr, "  /clear          Clear session context and approvals")
	fmt.Fprintln(os.Stderr, "  /bye            Exit")
	fmt.Fprintln(os.Stderr, "  /?, /help       Help for a command")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Keyboard Shortcuts:")
	fmt.Fprintln(os.Stderr, "  Ctrl+O          Expand last tool output")
	fmt.Fprintln(os.Stderr, "")
}

// handleSetCommand verarbeitet das /set Kommando.
// Gibt true zurück wenn der Scanner Readline fortsetzen soll.
func handleSetCommand(cmd *cobra.Command, args []string, modelName string, scanner *readline.Instance,
	wordWrap *bool, think **api.ThinkValue, format *string, system *string,
	options map[string]any, messages *[]api.Message) {

	if len(args) <= 1 {
		fmt.Println("Usage: /set <parameter|system|history|format|wordwrap|think|verbose> [value]")
		return
	}

	switch args[1] {
	case "history":
		scanner.HistoryEnable()
	case "nohistory":
		scanner.HistoryDisable()
	case "wordwrap":
		*wordWrap = true
		fmt.Println("Set 'wordwrap' mode.")
	case "nowordwrap":
		*wordWrap = false
		fmt.Println("Set 'nowordwrap' mode.")
	case "verbose":
		handleSetVerbose(cmd, true)
	case "quiet":
		handleSetVerbose(cmd, false)
	case "think":
		handleSetThink(cmd, args, modelName, think)
	case "nothink":
		handleSetNoThink(cmd, modelName, think)
	case "format":
		handleSetFormat(args, format)
	case "noformat":
		*format = ""
		fmt.Println("Disabled format.")
	case "parameter":
		handleSetParameter(args, options)
	case "system":
		handleSetSystem(args, system, messages)
	default:
		fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
	}
}

// handleSetVerbose setzt den Verbose-Modus.
func handleSetVerbose(cmd *cobra.Command, verbose bool) {
	value := "false"
	mode := "quiet"
	if verbose {
		value = "true"
		mode = "verbose"
	}
	if err := cmd.Flags().Set("verbose", value); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	fmt.Printf("Set '%s' mode.\n", mode)
}

// handleSetThink aktiviert den Thinking-Modus.
func handleSetThink(cmd *cobra.Command, args []string, modelName string, think **api.ThinkValue) {
	thinkValue := api.ThinkValue{Value: true}
	var maybeLevel string
	if len(args) > 2 {
		maybeLevel = args[2]
	}
	if maybeLevel != "" {
		thinkValue.Value = maybeLevel
	}
	*think = &thinkValue

	// Prüfen ob Modell Thinking unterstützt
	warnIfNoThinkingSupport(cmd, modelName)

	if maybeLevel != "" {
		fmt.Printf("Set 'think' mode to '%s'.\n", maybeLevel)
	} else {
		fmt.Println("Set 'think' mode.")
	}
}

// handleSetNoThink deaktiviert den Thinking-Modus.
func handleSetNoThink(cmd *cobra.Command, modelName string, think **api.ThinkValue) {
	*think = &api.ThinkValue{Value: false}
	warnIfNoThinkingSupport(cmd, modelName)
	fmt.Println("Set 'nothink' mode.")
}

// warnIfNoThinkingSupport warnt falls Modell kein Thinking unterstützt.
func warnIfNoThinkingSupport(cmd *cobra.Command, modelName string) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}
	resp, err := client.Show(cmd.Context(), &api.ShowRequest{Model: modelName})
	if err != nil {
		return
	}
	if !slices.Contains(resp.Capabilities, model.CapabilityThinking) {
		fmt.Fprintf(os.Stderr, "warning: model %q does not support thinking output\n", modelName)
	}
}

// handleSetFormat setzt das Ausgabeformat.
func handleSetFormat(args []string, format *string) {
	if len(args) < 3 || args[2] != "json" {
		fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
	} else {
		*format = args[2]
		fmt.Printf("Set format to '%s' mode.\n", args[2])
	}
}

// handleSetParameter setzt einen Modell-Parameter.
func handleSetParameter(args []string, options map[string]any) {
	if len(args) < 4 {
		fmt.Println("Usage: /set parameter <name> <value>")
		return
	}
	params := args[3:]
	fp, err := api.FormatParams(map[string][]string{args[2]: params})
	if err != nil {
		fmt.Printf("Couldn't set parameter: %q\n", err)
		return
	}
	fmt.Printf("Set parameter '%s' to '%s'\n", args[2], strings.Join(params, ", "))
	options[args[2]] = fp[args[2]]
}

// handleSetSystem setzt die System-Nachricht.
func handleSetSystem(args []string, system *string, messages *[]api.Message) {
	if len(args) < 3 {
		fmt.Println("Usage: /set system <message>")
		return
	}

	*system = strings.Join(args[2:], " ")
	newMessage := api.Message{Role: "system", Content: *system}
	if len(*messages) > 0 && (*messages)[len(*messages)-1].Role == "system" {
		(*messages)[len(*messages)-1] = newMessage
	} else {
		*messages = append(*messages, newMessage)
	}
	fmt.Println("Set system message.")
}

// handleLoadCommand verarbeitet das /load Kommando.
// Gibt den neuen Modellnamen zurück falls erfolgreich, sonst den alten.
func handleLoadCommand(cmd *cobra.Command, args []string, currentModel string, think *api.ThinkValue,
	messages *[]api.Message, approval *agent.ApprovalManager) string {

	if len(args) != 2 {
		fmt.Println("Usage: /load <modelname>")
		return currentModel
	}

	newModelName := args[1]
	fmt.Printf("Loading model '%s'\n", newModelName)

	// Progress-Spinner erstellen
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	// Client holen
	client, err := api.ClientFromEnvironment()
	if err != nil {
		p.StopAndClear()
		fmt.Println("error: couldn't connect to ollama server")
		return currentModel
	}

	// Prüfen ob Modell existiert und Info holen
	info, err := client.Show(cmd.Context(), &api.ShowRequest{Model: newModelName})
	if err != nil {
		p.StopAndClear()
		if strings.Contains(err.Error(), "not found") {
			fmt.Printf("Couldn't find model '%s'\n", newModelName)
		} else {
			fmt.Printf("error: %v\n", err)
		}
		return currentModel
	}

	// Für Cloud-Modelle kein Preload nötig
	if info.RemoteHost == "" {
		if !preloadModel(cmd, client, newModelName, think, p) {
			return currentModel
		}
	}

	p.StopAndClear()
	*messages = []api.Message{}
	approval.Reset()
	return newModelName
}

// preloadModel lädt ein Modell vor. Gibt false bei Fehler zurück.
func preloadModel(cmd *cobra.Command, client *api.Client, modelName string,
	think *api.ThinkValue, p *progress.Progress) bool {

	req := &api.GenerateRequest{
		Model: modelName,
		Think: think,
	}
	err := client.Generate(cmd.Context(), req, func(r api.GenerateResponse) error {
		return nil
	})
	if err != nil {
		p.StopAndClear()
		if strings.Contains(err.Error(), "not found") {
			fmt.Printf("Couldn't find model '%s'\n", modelName)
		} else if strings.Contains(err.Error(), "does not support thinking") {
			fmt.Printf("error: %v\n", err)
		} else {
			fmt.Printf("error loading model: %v\n", err)
		}
		return false
	}
	return true
}

// handleSaveCommand verarbeitet das /save Kommando.
func handleSaveCommand(cmd *cobra.Command, args []string, modelName string, options map[string]any, messages []api.Message) {
	if len(args) != 2 {
		fmt.Println("Usage: /save <modelname>")
		return
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Println("error: couldn't connect to ollama server")
		return
	}

	req := &api.CreateRequest{
		Model:      args[1],
		From:       modelName,
		Parameters: options,
		Messages:   messages,
	}
	fn := func(resp api.ProgressResponse) error { return nil }
	err = client.Create(cmd.Context(), req, fn)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	fmt.Printf("Created new model '%s'\n", args[1])
}
