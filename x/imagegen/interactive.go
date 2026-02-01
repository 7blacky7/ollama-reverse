// interactive.go - Interaktiver REPL-Modus fuer Bildgenerierung.
//
// Dieses Modul enthaelt:
// - runInteractive REPL-Schleife
// - Kommando-Handling (/set, /show, /help, /bye)
// - Einstellungs-Verwaltung

package imagegen

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
)

// runInteractive runs an interactive REPL for image generation.
func runInteractive(cmd *cobra.Command, modelName string, keepAlive *api.Duration, opts ImageGenOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Preload the model with the specified keepalive
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	preloadReq := &api.GenerateRequest{
		Model:     modelName,
		KeepAlive: keepAlive,
	}
	if err := client.Generate(cmd.Context(), preloadReq, func(resp api.GenerateResponse) error {
		return nil
	}); err != nil {
		p.StopAndClear()
		return fmt.Errorf("failed to load model: %w", err)
	}
	p.StopAndClear()

	scanner, err := readline.New(readline.Prompt{
		Prompt:      ">>> ",
		Placeholder: "Describe an image to generate (/help for commands)",
	})
	if err != nil {
		return err
	}

	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}

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
			continue
		case err != nil:
			return err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Handle commands
		switch {
		case strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/?"), strings.HasPrefix(line, "/help"):
			printInteractiveHelp()
			continue
		case strings.HasPrefix(line, "/set "):
			if err := handleSetCommand(line[5:], &opts); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		case strings.HasPrefix(line, "/show"):
			printCurrentSettings(opts)
			continue
		case strings.HasPrefix(line, "/"):
			// Check if it's a file path, not a command
			args := strings.Fields(line)
			isFile := false
			for _, f := range extractFileNames(line) {
				if strings.HasPrefix(f, args[0]) {
					isFile = true
					break
				}
			}
			if !isFile {
				fmt.Fprintf(os.Stderr, "Unknown command: %s (try /help)\n", args[0])
				continue
			}
		}

		// Generate image
		if err := generateInteractiveImage(cmd, client, modelName, line, keepAlive, opts); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		fmt.Println()
	}
}

// generateInteractiveImage generates an image in interactive mode.
func generateInteractiveImage(cmd *cobra.Command, client *api.Client, modelName, line string, keepAlive *api.Duration, opts ImageGenOptions) error {
	// Extract any image paths from the input
	prompt, images, err := extractFileData(line)
	if err != nil {
		return err
	}

	// Generate image with current options
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Images: images,
		Width:  int32(opts.Width),
		Height: int32(opts.Height),
		Steps:  int32(opts.Steps),
	}
	if opts.Seed != 0 {
		req.Options = map[string]any{"seed": opts.Seed}
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}

	// Show loading spinner until generation starts
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var stepBar *progress.StepBar
	var imageBase64 string

	err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
		// Handle progress updates using structured fields
		if resp.Total > 0 {
			if stepBar == nil {
				spinner.Stop()
				stepBar = progress.NewStepBar("Generating", int(resp.Total))
				p.Add("", stepBar)
			}
			stepBar.Set(int(resp.Completed))
		}

		// Handle final response with image data
		if resp.Done && resp.Image != "" {
			imageBase64 = resp.Image
		}

		return nil
	})

	p.StopAndClear()
	if err != nil {
		return err
	}

	// Save image to current directory with descriptive name
	if imageBase64 != "" {
		// Decode base64 image data
		imageData, err := base64.StdEncoding.DecodeString(imageBase64)
		if err != nil {
			return fmt.Errorf("error decoding image: %w", err)
		}

		// Create filename from prompt (sanitized)
		safeName := sanitizeFilename(line)
		if len(safeName) > 50 {
			safeName = safeName[:50]
		}
		timestamp := time.Now().Format("20060102-150405")
		filename := fmt.Sprintf("%s-%s.png", safeName, timestamp)

		if err := os.WriteFile(filename, imageData, 0o644); err != nil {
			return fmt.Errorf("error saving image: %w", err)
		}

		displayImageInTerminal(filename)
		fmt.Printf("Image saved to: %s\n", filename)
	}

	return nil
}

// printInteractiveHelp prints help for interactive mode commands.
// TODO: reconcile /set commands with /set parameter in text gen REPL (cmd/cmd.go)
func printInteractiveHelp() {
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  /set width <n>     Set image width")
	fmt.Fprintln(os.Stderr, "  /set height <n>    Set image height")
	fmt.Fprintln(os.Stderr, "  /set steps <n>     Set denoising steps")
	fmt.Fprintln(os.Stderr, "  /set seed <n>      Set random seed")
	fmt.Fprintln(os.Stderr, "  /set negative <s>  Set negative prompt")
	fmt.Fprintln(os.Stderr, "  /show              Show current settings")
	fmt.Fprintln(os.Stderr, "  /bye               Exit")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Or type a prompt to generate an image.")
	fmt.Fprintln(os.Stderr)
}

// printCurrentSettings prints the current image generation settings.
func printCurrentSettings(opts ImageGenOptions) {
	fmt.Fprintf(os.Stderr, "Current settings:\n")
	fmt.Fprintf(os.Stderr, "  width:    %d\n", opts.Width)
	fmt.Fprintf(os.Stderr, "  height:   %d\n", opts.Height)
	fmt.Fprintf(os.Stderr, "  steps:    %d\n", opts.Steps)
	fmt.Fprintf(os.Stderr, "  seed:     %d (0=random)\n", opts.Seed)
	if opts.NegativePrompt != "" {
		fmt.Fprintf(os.Stderr, "  negative: %s\n", opts.NegativePrompt)
	}
	fmt.Fprintln(os.Stderr)
}

// handleSetCommand handles /set commands to change options.
func handleSetCommand(args string, opts *ImageGenOptions) error {
	parts := strings.SplitN(args, " ", 2)
	if len(parts) < 2 {
		return fmt.Errorf("usage: /set <option> <value>")
	}

	key := strings.ToLower(parts[0])
	value := strings.TrimSpace(parts[1])

	switch key {
	case "width", "w":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("width must be a positive integer")
		}
		opts.Width = v
		fmt.Fprintf(os.Stderr, "Set width to %d\n", v)
	case "height", "h":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("height must be a positive integer")
		}
		opts.Height = v
		fmt.Fprintf(os.Stderr, "Set height to %d\n", v)
	case "steps", "s":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("steps must be a positive integer")
		}
		opts.Steps = v
		fmt.Fprintf(os.Stderr, "Set steps to %d\n", v)
	case "seed":
		v, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("seed must be an integer")
		}
		opts.Seed = v
		fmt.Fprintf(os.Stderr, "Set seed to %d\n", v)
	case "negative", "neg", "n":
		opts.NegativePrompt = value
		if value == "" {
			fmt.Fprintln(os.Stderr, "Cleared negative prompt")
		} else {
			fmt.Fprintf(os.Stderr, "Set negative prompt to: %s\n", value)
		}
	default:
		return fmt.Errorf("unknown option: %s (try /help)", key)
	}
	return nil
}
