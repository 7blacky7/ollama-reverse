// options.go - Optionen und Flag-Definitionen fuer die Bildgenerierung.
//
// Dieses Modul enthaelt:
// - ImageGenOptions Struktur fuer Generierungsparameter
// - Standardwerte und Flag-Registrierung
// - Hilfstext-Formatierung

package imagegen

import (
	"github.com/spf13/cobra"
)

// ImageGenOptions holds options for image generation.
// These can be set via environment variables or interactive commands.
type ImageGenOptions struct {
	Width          int
	Height         int
	Steps          int
	Seed           int
	NegativePrompt string
}

// DefaultOptions returns the default image generation options.
func DefaultOptions() ImageGenOptions {
	return ImageGenOptions{
		Width:  1024,
		Height: 1024,
		Steps:  0, // 0 means model default
		Seed:   0, // 0 means random
	}
}

// RegisterFlags adds image generation flags to the given command.
// Flags are hidden since they only apply to image generation models.
func RegisterFlags(cmd *cobra.Command) {
	cmd.Flags().Int("width", 1024, "Image width")
	cmd.Flags().Int("height", 1024, "Image height")
	cmd.Flags().Int("steps", 0, "Denoising steps (0 = model default)")
	cmd.Flags().Int("seed", 0, "Random seed (0 for random)")
	cmd.Flags().String("negative", "", "Negative prompt")
	// Hide from main flags section - shown in separate section via AppendFlagsDocs
	cmd.Flags().MarkHidden("width")
	cmd.Flags().MarkHidden("height")
	cmd.Flags().MarkHidden("steps")
	cmd.Flags().MarkHidden("seed")
	cmd.Flags().MarkHidden("negative")
}

// AppendFlagsDocs appends image generation flags documentation to the command's usage template.
func AppendFlagsDocs(cmd *cobra.Command) {
	usage := `
Image Generation Flags (experimental):
      --width int      Image width
      --height int     Image height
      --steps int      Denoising steps
      --seed int       Random seed
      --negative str   Negative prompt
`
	cmd.SetUsageTemplate(cmd.UsageTemplate() + usage)
}

// ParseOptionsFromFlags extracts ImageGenOptions from cobra command flags.
func ParseOptionsFromFlags(cmd *cobra.Command) ImageGenOptions {
	opts := DefaultOptions()
	if cmd != nil && cmd.Flags() != nil {
		if v, err := cmd.Flags().GetInt("width"); err == nil && v > 0 {
			opts.Width = v
		}
		if v, err := cmd.Flags().GetInt("height"); err == nil && v > 0 {
			opts.Height = v
		}
		if v, err := cmd.Flags().GetInt("steps"); err == nil && v > 0 {
			opts.Steps = v
		}
		if v, err := cmd.Flags().GetInt("seed"); err == nil && v != 0 {
			opts.Seed = v
		}
		if v, err := cmd.Flags().GetString("negative"); err == nil && v != "" {
			opts.NegativePrompt = v
		}
	}
	return opts
}
