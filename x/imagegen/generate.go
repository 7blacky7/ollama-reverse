// generate.go - Bildgenerierungs-Logik fuer CLI.
//
// Dieses Modul enthaelt:
// - RunCLI Einstiegspunkt
// - generateImageWithOptions fuer One-Shot Generierung
// - Progress-Anzeige und Speicherung

package imagegen

import (
	"encoding/base64"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
)

// RunCLI handles the CLI for image generation models.
// Returns true if it handled the request, false if the caller should continue with normal flow.
// Supports flags: --width, --height, --steps, --seed, --negative
// Image paths can be included in the prompt and will be extracted automatically.
func RunCLI(cmd *cobra.Command, name string, prompt string, interactive bool, keepAlive *api.Duration) error {
	opts := ParseOptionsFromFlags(cmd)

	if interactive {
		return runInteractive(cmd, name, keepAlive, opts)
	}

	// One-shot generation
	return generateImageWithOptions(cmd, name, prompt, keepAlive, opts)
}

// generateImageWithOptions generates an image with the given options.
func generateImageWithOptions(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration, opts ImageGenOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Extract any image paths from the prompt
	prompt, images, err := extractFileData(prompt)
	if err != nil {
		return err
	}

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

	if imageBase64 != "" {
		// Decode base64 and save to CWD
		imageData, err := base64.StdEncoding.DecodeString(imageBase64)
		if err != nil {
			return fmt.Errorf("failed to decode image: %w", err)
		}

		// Create filename from prompt
		safeName := sanitizeFilename(prompt)
		if len(safeName) > 50 {
			safeName = safeName[:50]
		}
		timestamp := time.Now().Format("20060102-150405")
		filename := fmt.Sprintf("%s-%s.png", safeName, timestamp)

		if err := os.WriteFile(filename, imageData, 0o644); err != nil {
			return fmt.Errorf("failed to save image: %w", err)
		}

		displayImageInTerminal(filename)
		fmt.Printf("Image saved to: %s\n", filename)
	}

	return nil
}
