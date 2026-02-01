//go:build mlx

// generate.go - Hauptmodul für die Text-Generierung mit MLX
// Enthält: generate-Funktion, input/output Structs
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Dedicated stream for generation (like mlx-lm's generation_stream)
var generationStream *mlx.Stream

// withStream runs fn with the generation stream as default
func withStream(fn func()) {
	// Lazy initialization of generationStream
	if generationStream == nil {
		generationStream = mlx.NewStream()
	}
	orig := mlx.GetDefaultStream()
	mlx.SetDefaultStream(generationStream)
	fn()
	mlx.SetDefaultStream(orig)
}

type Model interface {
	Tokenizer() *tokenizer.Tokenizer
	VocabSize() int32
	NewCache(maxSeqLen int32) []cache.Cache
	Forward(input *mlx.Array, caches []cache.Cache) *mlx.Array
}

// ChatModel is an optional interface for models that support chat formatting
type ChatModel interface {
	FormatPrompt(prompt string) string
}

// MultimodalModel is for models that support image input
type MultimodalModel interface {
	Model
	FormatPromptWithImage(prompt string) string
	ExpandImageTokens(tokens []int32) []int32
	ForwardWithImage(tokens *mlx.Array, image *mlx.Array, caches []cache.Cache) *mlx.Array
	ImageSize() int32 // Returns expected image size for preprocessing
}

// ImageLoader loads and preprocesses an image for multimodal models
// Returns nil if path is empty
type ImageLoader func(path string, imageSize int32) (*mlx.Array, error)

type input struct {
	Prompt       string
	Image        *mlx.Array // Optional preprocessed image for multimodal models
	MaxTokens    int
	Temperature  float32
	TopP         float32
	TopK         int
	WiredLimitGB int // Metal wired memory limit in GB (default 32)
}

type output struct {
	Text          string
	Done          bool
	PrefillTokSec float64
	GenTokSec     float64
}

// generate führt die Text-Generierung mit dem Modell durch
func generate(ctx context.Context, m Model, in input, cb func(output)) error {
	mlx.EnableCompile()
	wiredLimit := in.WiredLimitGB
	if wiredLimit <= 0 {
		wiredLimit = 32 // default 32GB
	}
	mlx.MetalSetWiredLimit(uint64(wiredLimit) << 30)

	temp := in.Temperature
	if temp < 0 {
		temp = 0.7
	}

	tok := m.Tokenizer()
	dec := NewDecoder(m, temp, in.TopK, in.TopP)

	// Apply chat template - use image template if we have an image
	prompt := in.Prompt
	var tokens []int32
	if mm, ok := m.(MultimodalModel); ok && in.Image != nil {
		prompt = mm.FormatPromptWithImage(prompt)
		tokens = tok.Encode(prompt, true)
		tokens = mm.ExpandImageTokens(tokens) // Expand <start_of_image> to 256 image tokens
		dec.SetImage(in.Image)
	} else if cm, ok := m.(ChatModel); ok {
		prompt = cm.FormatPrompt(prompt)
		tokens = tok.Encode(prompt, true)
	} else {
		tokens = tok.Encode(prompt, true)
	}

	prefillStart := time.Now()
	prefillTokens := dec.prefill(tokens)
	// Prefill measurement should include time to first token (like mlx-lm)
	// Step() waits for prefill to complete and returns first token
	firstToken := dec.step()
	prefillTokSec := float64(prefillTokens) / time.Since(prefillStart).Seconds()

	genStart := time.Now()
	maxTokens := max(in.MaxTokens, 100)
	var genTokens int

	// UTF-8 streamer to handle partial multi-byte characters
	streamer := &utf8Streamer{}

	// Handle first token
	genTokens++
	if tok.IsEOS(firstToken) {
		cb(output{Done: true, PrefillTokSec: prefillTokSec, GenTokSec: 0})
		return nil
	}
	if text := streamer.Write(tok.Decode([]int32{firstToken})); text != "" {
		cb(output{Text: text})
	}

	for n := 1; n < maxTokens; n++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		token := dec.step()
		genTokens++

		if tok.IsEOS(token) {
			break
		}
		if text := streamer.Write(tok.Decode([]int32{token})); text != "" {
			cb(output{Text: text})
		}

		if n%256 == 0 {
			mlx.ClearCache()
		}
	}

	// Flush any remaining buffered bytes
	if text := streamer.Flush(); text != "" {
		cb(output{Text: text})
	}

	fmt.Printf("\nPeak memory: %.2fGB\n", float64(mlx.MetalGetPeakMemory())/(1<<30))
	cb(output{Done: true, PrefillTokSec: prefillTokSec,
		GenTokSec: float64(genTokens) / time.Since(genStart).Seconds()})
	return nil
}
