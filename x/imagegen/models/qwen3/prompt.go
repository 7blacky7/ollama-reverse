//go:build mlx

// Modul: prompt.go
// Beschreibung: Prompt-Encoding und Chat-Template-Funktionen für den Qwen3 Text-Encoder.
// Enthält: ApplyChatTemplate, EncodePrompt, EncodePromptWithLayers.

package qwen3

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// ApplyChatTemplate wraps prompt in Qwen3 chat format.
// If think is true, adds the <think></think> block after the assistant tag
// (matches tokenizer.apply_chat_template with enable_thinking=False in Python).
func ApplyChatTemplate(prompt string, think bool) string {
	base := "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	if think {
		return base + "<think>\n\n</think>\n\n"
	}
	return base
}

// EncodePrompt encodes a text prompt using the tokenizer and encoder.
// If think is true, includes the <think></think> block in the chat template.
func (te *TextEncoder) EncodePrompt(tok *tokenizer.Tokenizer, prompt string, maxLen int, think bool) (*mlx.Array, *mlx.Array) {
	formattedPrompt := ApplyChatTemplate(prompt, think)

	tokens := tok.Encode(formattedPrompt, false)

	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
	}

	maskData := make([]float32, maxLen)
	for i := 0; i < len(tokens); i++ {
		maskData[i] = 1.0
	}

	// Get PAD token (different from EOS for Qwen3)
	padToken := tok.PAD()
	if padToken < 0 {
		padToken = tok.EOS() // fallback
	}

	paddedTokens := make([]int32, maxLen)
	copy(paddedTokens, tokens)
	for i := len(tokens); i < maxLen; i++ {
		paddedTokens[i] = padToken
	}

	tokensArr := mlx.NewArrayInt32(paddedTokens, []int32{1, int32(maxLen)})
	maskArr := mlx.NewArray(maskData, []int32{1, int32(maxLen)})

	// Build combined causal + PAD mask [L, L]
	// mask[i,j] = 0 if (j <= i AND valid[j]) else -inf
	L := int32(maxLen)
	validLen := int32(len(tokens))
	combinedMaskData := make([]float32, L*L)
	negInf := float32(-1e9)
	for i := int32(0); i < L; i++ {
		for j := int32(0); j < L; j++ {
			idx := i*L + j
			if j <= i && j < validLen {
				combinedMaskData[idx] = 0
			} else {
				combinedMaskData[idx] = negInf
			}
		}
	}
	maskMat := mlx.NewArray(combinedMaskData, []int32{L, L})

	embeddings := te.Forward(tokensArr, maskMat, "")

	return embeddings, maskArr
}

// EncodePromptWithLayers encodes a text prompt and returns embeddings from specified layers.
// Used by Flux2 which concatenates embeddings from multiple intermediate layers.
// If think is true, includes the <think></think> block in the chat template.
// Returns embeddings and padded sequence length.
func (te *TextEncoder) EncodePromptWithLayers(tok *tokenizer.Tokenizer, prompt string, maxLen int, layerIndices []int, think bool) (*mlx.Array, int32) {
	formattedPrompt := ApplyChatTemplate(prompt, think)
	tokens := tok.Encode(formattedPrompt, false)

	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
	}

	// Pad to maxLen
	padToken := tok.PAD()
	if padToken < 0 {
		padToken = tok.EOS() // fallback
	}
	padded := make([]int32, maxLen)
	copy(padded, tokens)
	for i := len(tokens); i < maxLen; i++ {
		padded[i] = padToken
	}
	tokensArr := mlx.NewArrayInt32(padded, []int32{1, int32(maxLen)})

	// Build combined causal + PAD mask [L, L]
	// mask[i,j] = 0 if (j <= i AND valid[j]) else -inf
	// This combines causal masking with PAD token masking
	L := int32(maxLen)
	validLen := int32(len(tokens))
	maskData := make([]float32, L*L)
	negInf := float32(-1e9)
	for i := int32(0); i < L; i++ {
		for j := int32(0); j < L; j++ {
			idx := i*L + j
			if j <= i && j < validLen {
				maskData[idx] = 0 // allowed: causal OK and not PAD
			} else {
				maskData[idx] = negInf // blocked: future or PAD
			}
		}
	}
	maskMat := mlx.NewArray(maskData, []int32{L, L})

	layerOutputs := te.ForwardWithLayerOutputs(tokensArr, layerIndices, maskMat, "")

	// Concatenate layer outputs along the hidden dimension
	// Each output is [B, L, hidden_dim], result is [B, L, num_layers * hidden_dim]
	embeddings := mlx.Concatenate(layerOutputs, 2)

	// Return embeddings and padded length
	return embeddings, int32(maxLen)
}
