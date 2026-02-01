// Modul: quantize.go
// Beschreibung: Logik zur Bestimmung, welche Tensoren quantisiert werden sollen.
// Enthält: ShouldQuantize, ShouldQuantizeTensor.

package create

import (
	"strings"
)

// ShouldQuantize returns true if a tensor should be quantized.
// For image gen models (component non-empty): quantizes linear weights, skipping VAE, embeddings, norms.
// For LLM models (component empty): quantizes linear weights, skipping embeddings, norms, and small tensors.
func ShouldQuantize(name, component string) bool {
	// Image gen specific: skip VAE entirely
	if component == "vae" {
		return false
	}

	// Skip embeddings
	if strings.Contains(name, "embed") {
		return false
	}

	// Skip layer norms and RMS norms
	if strings.Contains(name, "norm") || strings.Contains(name, "ln_") || strings.Contains(name, "layernorm") {
		return false
	}

	// Skip biases
	if strings.HasSuffix(name, ".bias") {
		return false
	}

	// Only quantize weights
	return strings.HasSuffix(name, ".weight")
}

// ShouldQuantizeTensor returns true if a tensor should be quantized based on name and shape.
// This is a more detailed check that also considers tensor dimensions.
func ShouldQuantizeTensor(name string, shape []int32) bool {
	// Use basic name-based check first
	if !ShouldQuantize(name, "") {
		return false
	}

	// Only quantize 2D tensors (linear layers) - skip 1D (biases, norms) and higher-D (convolutions if any)
	if len(shape) != 2 {
		return false
	}

	// Skip small tensors (less than 1024 elements) - not worth quantizing
	if len(shape) >= 2 && int64(shape[0])*int64(shape[1]) < 1024 {
		return false
	}

	// MLX quantization requires last dimension to be divisible by group size (32)
	if shape[len(shape)-1]%32 != 0 {
		return false
	}

	return true
}
