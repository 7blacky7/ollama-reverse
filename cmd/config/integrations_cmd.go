// Package config - Launch Command für Integrationen
//
// Cobra Command für "ollama launch":
// - LaunchCmd: Hauptkommando zum Starten von Integrationen
// - Unterstützt --model und --config Flags
// - Interaktive Auswahl wenn keine Parameter angegeben
package config

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// LaunchCmd gibt das Cobra Command zum Starten von Integrationen zurück
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error) *cobra.Command {
	var modelFlag string
	var configFlag bool

	cmd := &cobra.Command{
		Use:   "launch [INTEGRATION]",
		Short: "Launch an integration with Ollama",
		Long: `Launch an integration configured with Ollama models.

Supported integrations:
  claude    Claude Code
  codex     Codex
  droid     Droid
  opencode  OpenCode
  openclaw  OpenClaw (aliases: clawdbot, moltbot)

Examples:
  ollama launch
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch droid --config (does not auto-launch)`,
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runLaunchCommand(cmd, args, modelFlag, configFlag)
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	return cmd
}

func runLaunchCommand(cmd *cobra.Command, args []string, modelFlag string, configFlag bool) error {
	name, err := resolveIntegrationName(args)
	if err != nil {
		return err
	}

	r, ok := integrations[strings.ToLower(name)]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	if !configFlag && modelFlag == "" {
		if config, err := loadIntegration(name); err == nil && len(config.Models) > 0 {
			return runIntegration(name, config.Models[0])
		}
	}

	models, err := resolveModels(cmd, name, modelFlag)
	if err != nil {
		return err
	}

	if err := configureEditor(r, models); err != nil {
		return err
	}

	if err := saveIntegration(name, models); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}

	if editor, isEditor := r.(Editor); isEditor {
		if err := editor.Edit(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
		printEditorSuccess(r, models)
	}

	return handleLaunch(name, r, models, configFlag)
}

func resolveIntegrationName(args []string) (string, error) {
	if len(args) > 0 {
		return args[0], nil
	}
	name, err := selectIntegration()
	if errors.Is(err, errCancelled) {
		return "", nil
	}
	return name, err
}

func resolveModels(cmd *cobra.Command, name, modelFlag string) ([]string, error) {
	if modelFlag != "" {
		models := []string{modelFlag}
		if existing, err := loadIntegration(name); err == nil && len(existing.Models) > 0 {
			for _, m := range existing.Models {
				if m != modelFlag {
					models = append(models, m)
				}
			}
		}
		return models, nil
	}

	models, err := selectModels(cmd.Context(), name, "")
	if errors.Is(err, errCancelled) {
		return nil, nil
	}
	return models, err
}

func configureEditor(r Runner, models []string) error {
	editor, isEditor := r.(Editor)
	if !isEditor {
		return nil
	}

	paths := editor.Paths()
	if len(paths) == 0 {
		return nil
	}

	fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", r)
	for _, p := range paths {
		fmt.Fprintf(os.Stderr, "  %s\n", p)
	}
	fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

	if ok, _ := confirmPrompt("Proceed?"); !ok {
		return nil
	}
	return nil
}

func printEditorSuccess(r Runner, models []string) {
	if len(models) == 1 {
		fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], r)
	} else {
		fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), r, models[0])
	}
}

func handleLaunch(name string, r Runner, models []string, configFlag bool) error {
	if configFlag {
		if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", r)); launch {
			return runIntegration(name, models[0])
		}
		fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start with %s\n", strings.ToLower(name), models[0])
		return nil
	}
	return runIntegration(name, models[0])
}
