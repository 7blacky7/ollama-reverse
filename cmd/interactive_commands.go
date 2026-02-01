// interactive_commands.go - Kommando-Handler fuer den interaktiven Modus
// Verarbeitet /set, /load, /save, /clear und /help Befehle
package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/types/errtypes"
)

// commandResult repraesentiert das Ergebnis eines Kommando-Handlers
type commandResult int

const (
	commandDone commandResult = iota
	commandContinue
	commandMultiline
)

// handleLoadCommand verarbeitet den /load Befehl
func handleLoadCommand(cmd *cobra.Command, opts *runOptions, line string, thinkExplicitlySet *bool) error {
	args := strings.Fields(line)
	if len(args) != 2 {
		fmt.Println("Usage:\n  /load <modelname>")
		return nil
	}
	origOpts := opts.Copy()

	opts.Model = args[1]
	opts.Messages = []api.Message{}
	fmt.Printf("Loading model '%s'\n", opts.Model)
	var err error
	opts.Think, err = inferThinkingOption(nil, opts, *thinkExplicitlySet)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			fmt.Printf("Couldn't find model '%s'\n", opts.Model)
			*opts = origOpts.Copy()
			return nil
		}
		return err
	}
	if err := loadOrUnloadModel(cmd, opts); err != nil {
		if strings.Contains(err.Error(), "not found") {
			fmt.Printf("Couldn't find model '%s'\n", opts.Model)
			*opts = origOpts.Copy()
			return nil
		}
		if strings.Contains(err.Error(), "does not support thinking") {
			fmt.Printf("error: %v\n", err)
			return nil
		}
		return err
	}
	return nil
}

// handleSaveCommand verarbeitet den /save Befehl
func handleSaveCommand(cmd *cobra.Command, opts *runOptions, line string) error {
	args := strings.Fields(line)
	if len(args) != 2 {
		fmt.Println("Usage:\n  /save <modelname>")
		return nil
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Println("error: couldn't connect to ollama server")
		return err
	}

	req := NewCreateRequest(args[1], *opts)
	fn := func(resp api.ProgressResponse) error { return nil }
	err = client.Create(cmd.Context(), req, fn)
	if err != nil {
		if strings.Contains(err.Error(), errtypes.InvalidModelNameErrMsg) {
			fmt.Printf("error: The model name '%s' is invalid\n", args[1])
			return nil
		}
		return err
	}
	fmt.Printf("Created new model '%s'\n", args[1])
	return nil
}

// handleClearCommand verarbeitet den /clear Befehl
func handleClearCommand(opts *runOptions) {
	opts.Messages = []api.Message{}
	if opts.System != "" {
		newMessage := api.Message{Role: "system", Content: opts.System}
		opts.Messages = append(opts.Messages, newMessage)
	}
	fmt.Println("Cleared session context")
}

// handleSetCommand verarbeitet den /set Befehl
func handleSetCommand(cmd *cobra.Command, opts *runOptions, line string, scanner *readline.Instance, sb *strings.Builder, thinkExplicitlySet *bool) commandResult {
	args := strings.Fields(line)
	if len(args) <= 1 {
		usageSet()
		return commandDone
	}

	switch args[1] {
	case "history":
		scanner.HistoryEnable()
	case "nohistory":
		scanner.HistoryDisable()
	case "wordwrap":
		opts.WordWrap = true
		fmt.Println("Set 'wordwrap' mode.")
	case "nowordwrap":
		opts.WordWrap = false
		fmt.Println("Set 'nowordwrap' mode.")
	case "verbose":
		if err := cmd.Flags().Set("verbose", "true"); err != nil {
			fmt.Printf("error: %v\n", err)
		} else {
			fmt.Println("Set 'verbose' mode.")
		}
	case "quiet":
		if err := cmd.Flags().Set("verbose", "false"); err != nil {
			fmt.Printf("error: %v\n", err)
		} else {
			fmt.Println("Set 'quiet' mode.")
		}
	case "think":
		handleSetThink(cmd, opts, args, thinkExplicitlySet)
	case "nothink":
		handleSetNoThink(cmd, opts, thinkExplicitlySet)
	case "format":
		handleSetFormat(opts, args)
	case "noformat":
		opts.Format = ""
		fmt.Println("Disabled format.")
	case "parameter":
		return handleSetParameter(opts, args)
	case "system":
		return handleSetSystemCommand(opts, args, sb)
	default:
		fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
	}
	return commandDone
}

// handleSetThink verarbeitet /set think
func handleSetThink(cmd *cobra.Command, opts *runOptions, args []string, thinkExplicitlySet *bool) {
	thinkValue := api.ThinkValue{Value: true}
	var maybeLevel string
	if len(args) > 2 {
		maybeLevel = args[2]
	}
	if maybeLevel != "" {
		thinkValue.Value = maybeLevel
	}
	opts.Think = &thinkValue
	*thinkExplicitlySet = true
	if client, err := api.ClientFromEnvironment(); err == nil {
		ensureThinkingSupport(cmd.Context(), client, opts.Model)
	}
	if maybeLevel != "" {
		fmt.Printf("Set 'think' mode to '%s'.\n", maybeLevel)
	} else {
		fmt.Println("Set 'think' mode.")
	}
}

// handleSetNoThink verarbeitet /set nothink
func handleSetNoThink(cmd *cobra.Command, opts *runOptions, thinkExplicitlySet *bool) {
	opts.Think = &api.ThinkValue{Value: false}
	*thinkExplicitlySet = true
	if client, err := api.ClientFromEnvironment(); err == nil {
		ensureThinkingSupport(cmd.Context(), client, opts.Model)
	}
	fmt.Println("Set 'nothink' mode.")
}

// handleSetFormat verarbeitet /set format
func handleSetFormat(opts *runOptions, args []string) {
	if len(args) < 3 || args[2] != "json" {
		fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
	} else {
		opts.Format = args[2]
		fmt.Printf("Set format to '%s' mode.\n", args[2])
	}
}

// handleSetParameter verarbeitet /set parameter
func handleSetParameter(opts *runOptions, args []string) commandResult {
	if len(args) < 4 {
		usageParameters()
		return commandContinue
	}
	params := args[3:]
	fp, err := api.FormatParams(map[string][]string{args[2]: params})
	if err != nil {
		fmt.Printf("Couldn't set parameter: %q\n", err)
		return commandContinue
	}
	fmt.Printf("Set parameter '%s' to '%s'\n", args[2], strings.Join(params, ", "))
	opts.Options[args[2]] = fp[args[2]]
	return commandDone
}

// handleSetSystemCommand verarbeitet den /set system Befehl
func handleSetSystemCommand(opts *runOptions, args []string, sb *strings.Builder) commandResult {
	if len(args) < 3 {
		usageSet()
		return commandContinue
	}

	line := strings.Join(args[2:], " ")
	line, ok := strings.CutPrefix(line, `"""`)
	if !ok {
		// Single line system message
		sb.WriteString(line)
		setSystemMessage(opts, sb)
		return commandContinue
	}

	// Multi-line system message
	line, ok = strings.CutSuffix(line, `"""`)
	sb.WriteString(line)
	if ok {
		// Complete multiline in single input
		setSystemMessage(opts, sb)
		return commandContinue
	}

	return commandMultiline
}

// setSystemMessage setzt die System-Nachricht aus dem StringBuilder
func setSystemMessage(opts *runOptions, sb *strings.Builder) {
	opts.System = sb.String()
	newMessage := api.Message{Role: "system", Content: sb.String()}
	if len(opts.Messages) > 0 && opts.Messages[len(opts.Messages)-1].Role == "system" {
		opts.Messages[len(opts.Messages)-1] = newMessage
	} else {
		opts.Messages = append(opts.Messages, newMessage)
	}
	fmt.Println("Set system message.")
	sb.Reset()
}

// handleHelpCommand verarbeitet den /help und /? Befehl
func handleHelpCommand(line string, opts runOptions) {
	args := strings.Fields(line)
	if len(args) > 1 {
		switch args[1] {
		case "set", "/set":
			usageSet()
		case "show", "/show":
			usageShow()
		case "shortcut", "shortcuts":
			usageShortcuts()
		}
	} else {
		usage(opts)
	}
}
