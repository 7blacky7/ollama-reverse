// Modul: layer.go
// Beschreibung: Transformer-Layer fuer das DeepSeek2-Modell
// Hauptstrukturen:
//   - Layer: Einzelne Transformer-Schicht mit Attention und MLP

package deepseek2

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// Layer repraesentiert eine einzelne Transformer-Schicht
// bestehend aus Attention und MLP mit Pre-Normalization
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention     *Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     MLP
}

// Forward fuehrt den Vorwaertsdurchlauf einer Layer durch.
// Verwendet Pre-Normalization und Residual-Verbindungen.
func (t *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	return hiddenStates
}
