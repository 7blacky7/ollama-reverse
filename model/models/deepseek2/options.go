// Modul: options.go
// Beschreibung: Konfigurationsoptionen fuer das DeepSeek2-Modell
// Hauptstrukturen:
//   - Options: Enthaelt alle Modell-spezifischen Konfigurationsparameter
//   - applyRotaryPositionEmbeddings: Wendet RoPE auf Tensoren an

package deepseek2

import (
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

// Options enthaelt alle konfigurierbaren Parameter fuer das DeepSeek2-Modell
type Options struct {
	isMLA               bool
	numExpertsUsed      int
	numExperts          int
	normTopKProb        bool
	routedScalingFactor float32

	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	kqNopeHeadDim,
	qkHeadDim int
	qLoraRank int
	vHeadDim  int

	hiddenSize,
	numHeads,
	numKVHeads,
	originalContextLength int

	eps,
	ropeBase,
	ropeScale float32
	kqScale float64
}

// applyRotaryPositionEmbeddings wendet Rotary Position Embeddings (RoPE)
// auf den gegebenen Tensor t mit Positionsinformationen p an
func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.qkRopeHeadDim, o.ropeBase, 1./o.ropeScale,
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
		rope.WithAttentionFactor(float32(1.0/(1.0+0.1*math.Log(float64(o.ropeScale))))),
	)
}
