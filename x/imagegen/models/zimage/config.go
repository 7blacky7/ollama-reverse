//go:build mlx

// config.go - Konfigurationsstrukturen fuer Z-Image Bildgenerierung
// Enthaelt alle Optionen fuer die Bildgenerierung inkl. TeaCache und CFG.
package zimage

// GenerateConfig holds all options for image generation.
type GenerateConfig struct {
	Prompt         string
	NegativePrompt string                   // Empty = no CFG
	CFGScale       float32                  // Only used if NegativePrompt is set (default: 4.0)
	Width          int32                    // Image width (default: 1024)
	Height         int32                    // Image height (default: 1024)
	Steps          int                      // Denoising steps (default: 9 for turbo)
	Seed           int64                    // Random seed
	Progress       func(step, totalSteps int) // Optional progress callback
	CapturePath    string                   // GPU capture path (debug)

	// TeaCache options (timestep embedding aware caching)
	TeaCache          bool    // TeaCache is always enabled for faster inference
	TeaCacheThreshold float32 // Threshold for cache reuse (default: 0.1, lower = more aggressive)

	// Fused QKV (fuse Q/K/V projections into single matmul)
	FusedQKV bool // Enable fused QKV projection (default: false)
}

// applyDefaults sets default values for unset config fields.
func (cfg *GenerateConfig) applyDefaults() {
	if cfg.Width <= 0 {
		cfg.Width = 1024
	}
	if cfg.Height <= 0 {
		cfg.Height = 1024
	}
	if cfg.Steps <= 0 {
		cfg.Steps = 9 // Z-Image turbo default
	}
	if cfg.CFGScale <= 0 {
		cfg.CFGScale = 4.0
	}
	// TeaCache enabled by default
	cfg.TeaCache = true
	if cfg.TeaCacheThreshold <= 0 {
		cfg.TeaCacheThreshold = 0.15
	}
}
