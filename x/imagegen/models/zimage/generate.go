//go:build mlx

// generate.go - Oeffentliche Generate-Methoden fuer Z-Image
// Enthaelt alle oeffentlichen API-Methoden zur Bildgenerierung.
package zimage

import (
	"context"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Generate creates an image from a prompt.
func (m *Model) Generate(prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt: prompt,
		Width:  width,
		Height: height,
		Steps:  steps,
		Seed:   seed,
	})
}

// GenerateWithProgress creates an image with progress callback.
func (m *Model) GenerateWithProgress(prompt string, width, height int32, steps int, seed int64, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}

// GenerateWithCFG creates an image with classifier-free guidance.
func (m *Model) GenerateWithCFG(prompt, negativePrompt string, width, height int32, steps int, seed int64, cfgScale float32, progress func(step, totalSteps int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(context.Background(), &GenerateConfig{
		Prompt:         prompt,
		NegativePrompt: negativePrompt,
		CFGScale:       cfgScale,
		Width:          width,
		Height:         height,
		Steps:          steps,
		Seed:           seed,
		Progress:       progress,
	})
}

// GenerateFromConfig generates an image using the unified config struct.
func (m *Model) GenerateFromConfig(ctx context.Context, cfg *GenerateConfig) (*mlx.Array, error) {
	start := time.Now()
	result, err := m.generate(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if cfg.NegativePrompt != "" {
		fmt.Printf("Generated with CFG (scale=%.1f) in %.2fs (%d steps)\n", cfg.CFGScale, time.Since(start).Seconds(), cfg.Steps)
	} else {
		fmt.Printf("Generated in %.2fs (%d steps)\n", time.Since(start).Seconds(), cfg.Steps)
	}
	return result, nil
}

// GenerateImage implements runner.ImageModel interface.
func (m *Model) GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error) {
	return m.GenerateFromConfig(ctx, &GenerateConfig{
		Prompt:   prompt,
		Width:    width,
		Height:   height,
		Steps:    steps,
		Seed:     seed,
		Progress: progress,
	})
}
