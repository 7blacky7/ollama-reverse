// cmd_builders.go - Command-Builder Funktionen
// Hauptfunktionen: newRunCmd, newStopCmd, etc.
package cmd

import (
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/x/imagegen"
)

// newRunCmd - Erstellt den run Command
func newRunCmd() *cobra.Command {
	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().String("keepalive", "", "Duration to keep a model loaded (e.g. 5m)")
	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")
	runCmd.Flags().String("think", "", "Enable thinking mode: true/false or high/medium/low for supported models")
	runCmd.Flags().Lookup("think").NoOptDefVal = "true"
	runCmd.Flags().Bool("hidethinking", false, "Hide thinking output (if provided)")
	runCmd.Flags().Bool("truncate", false, "For embedding models: truncate inputs exceeding context length")
	runCmd.Flags().Int("dimensions", 0, "Truncate output embeddings to specified dimension")
	runCmd.Flags().Bool("experimental", false, "Enable experimental agent loop with tools")
	runCmd.Flags().Bool("experimental-yolo", false, "Skip all tool approval prompts (use with caution)")
	runCmd.Flags().Bool("experimental-websearch", false, "Enable web search tool in experimental mode")

	imagegen.RegisterFlags(runCmd)

	return runCmd
}

// newStopCmd - Erstellt den stop Command
func newStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "stop MODEL",
		Short:   "Stop a running model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    StopHandler,
	}
}
