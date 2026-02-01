// text_options.go - Konfigurationsoptionen für das Gemma3n Text-Modell
// Enthält: TextOptions Struct und Hilfsmethoden
package gemma3n

import (
	"cmp"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

// TextOptions enthält alle Konfigurationsparameter für das Text-Modell
type TextOptions struct {
	hiddenLayers            int
	hiddenSize              int
	hiddenSizePerLayerInput int
	numHeads, numKVHeads    int
	keyLength, valueLength  int
	sharedKeyValueLayers    int

	altupActiveIndex  int
	altupInputs       int
	altupCorrectScale bool

	eps           float32
	ropeBase      float32
	ropeBaseLocal float32
	ropeScale     float32

	slidingWindowPattern    []bool
	activationSparsityScale []float32
}

// altupActive gibt den aktiven Teil des Tensors zurück
func (o *TextOptions) altupActive(ctx ml.Context, t ml.Tensor) ml.Tensor {
	// t[:, :, o.altupActiveIndex]
	return t.Slice(ctx, 2, o.altupActiveIndex, o.altupActiveIndex+1, 1)
}

// headDim berechnet die Dimension pro Attention-Head
func (o *TextOptions) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

// isLocal prüft, ob der Layer ein lokaler (sliding window) Layer ist
func (o *TextOptions) isLocal(i int) bool {
	return o.slidingWindowPattern[i]
}

// applyRotaryPositionEmbeddings wendet RoPE auf den Tensor an
func (o TextOptions) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor, base float32) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.headDim(), base, 1./o.ropeScale, rope.WithTypeNeoX())
}
