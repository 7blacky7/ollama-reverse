// cmd_run.go - Run Command Handler
// Hauptfunktionen: RunHandler, StopHandler
package cmd

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
	xcmd "github.com/ollama/ollama/x/cmd"
	"github.com/ollama/ollama/x/imagegen"
)

// RunHandler - Haupthandler fuer den run Command
func RunHandler(cmd *cobra.Command, args []string) error {
	interactive := true

	opts := runOptions{
		Model:       args[0],
		WordWrap:    os.Getenv("TERM") == "xterm-256color",
		Options:     map[string]any{},
		ShowConnect: true,
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	thinkFlag := cmd.Flags().Lookup("think")
	if thinkFlag.Changed {
		thinkStr, err := cmd.Flags().GetString("think")
		if err != nil {
			return err
		}

		switch thinkStr {
		case "", "true":
			opts.Think = &api.ThinkValue{Value: true}
		case "false":
			opts.Think = &api.ThinkValue{Value: false}
		case "high", "medium", "low":
			opts.Think = &api.ThinkValue{Value: thinkStr}
		default:
			return fmt.Errorf("invalid value for --think: %q (must be true, false, high, medium, or low)", thinkStr)
		}
	} else {
		opts.Think = nil
	}

	hidethinking, err := cmd.Flags().GetBool("hidethinking")
	if err != nil {
		return err
	}
	opts.HideThinking = hidethinking

	keepAlive, err := cmd.Flags().GetString("keepalive")
	if err != nil {
		return err
	}
	if keepAlive != "" {
		d, err := time.ParseDuration(keepAlive)
		if err != nil {
			return err
		}
		opts.KeepAlive = &api.Duration{Duration: d}
	}

	prompts := args[1:]
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := readStdinContent()
		if err != nil {
			return err
		}
		if len(in) > 0 {
			prompts = append([]string{in}, prompts...)
		}
		opts.ShowConnect = false
		opts.WordWrap = false
		interactive = false
	}
	opts.Prompt = strings.Join(prompts, " ")
	if len(prompts) > 0 {
		interactive = false
	}
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		interactive = false
	}

	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	opts.WordWrap = !nowrap

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	name := args[0]

	info, err := getModelInfo(cmd, client, name)
	if err != nil {
		return err
	}

	opts.Think, err = inferThinkingOption(&info.Capabilities, &opts, thinkFlag.Changed)
	if err != nil {
		return err
	}

	opts.MultiModal = slices.Contains(info.Capabilities, model.CapabilityVision)

	if len(info.ProjectorInfo) != 0 {
		opts.MultiModal = true
	}
	for k := range info.ModelInfo {
		if strings.Contains(k, ".vision.") {
			opts.MultiModal = true
			break
		}
	}

	opts.ParentModel = info.Details.ParentModel

	// Embedding-Modell?
	if slices.Contains(info.Capabilities, model.CapabilityEmbedding) {
		return handleEmbeddingModel(cmd, name, opts)
	}

	// Image-Generation-Modell?
	if slices.Contains(info.Capabilities, model.CapabilityImage) {
		if opts.Prompt == "" && !interactive {
			return errors.New("image generation models require a prompt. Usage: ollama run " + name + " \"your prompt here\"")
		}
		return imagegen.RunCLI(cmd, name, opts.Prompt, interactive, opts.KeepAlive)
	}

	isExperimental, _ := cmd.Flags().GetBool("experimental")
	yoloMode, _ := cmd.Flags().GetBool("experimental-yolo")
	enableWebsearch, _ := cmd.Flags().GetBool("experimental-websearch")

	if interactive {
		if err := loadOrUnloadModel(cmd, &opts); err != nil {
			return handleAuthError(err)
		}

		for _, msg := range info.Messages {
			switch msg.Role {
			case "user":
				fmt.Printf(">>> %s\n", msg.Content)
			case "assistant":
				state := &displayResponseState{}
				displayResponse(msg.Content, opts.WordWrap, state)
				fmt.Println()
				fmt.Println()
			}
		}

		if isExperimental {
			return xcmd.GenerateInteractive(cmd, opts.Model, opts.WordWrap, opts.Options, opts.Think, opts.HideThinking, opts.KeepAlive, yoloMode, enableWebsearch)
		}

		return generateInteractive(cmd, opts)
	}
	return generate(cmd, opts)
}

// StopHandler - Stoppt ein laufendes Modell
func StopHandler(cmd *cobra.Command, args []string) error {
	opts := &runOptions{
		Model:     args[0],
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := loadOrUnloadModel(cmd, opts); err != nil {
		if strings.Contains(err.Error(), "not found") {
			return fmt.Errorf("couldn't find model \"%s\" to stop", args[0])
		}
		return err
	}
	return nil
}

// handleAuthError - Behandelt Authentifizierungsfehler
func handleAuthError(err error) error {
	var sErr api.AuthorizationError
	if errors.As(err, &sErr) && sErr.StatusCode == http.StatusUnauthorized {
		fmt.Printf("You need to be signed in to Ollama to run Cloud models.\n\n")
		if sErr.SigninURL != "" {
			fmt.Printf(ConnectInstructions, sErr.SigninURL)
		}
		return nil
	}
	return err
}
