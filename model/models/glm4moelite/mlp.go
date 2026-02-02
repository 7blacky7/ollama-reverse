// Package glm4moelite - MLP (Multi-Layer Perceptron) Komponenten
//
// Diese Datei enthaelt:
// - MLP Interface: Schnittstelle fuer Feed-Forward-Netzwerke
// - sparse: Mixture-of-Experts (MoE) Implementierung mit Routing
// - dense: Standard Feed-Forward Implementierung
//
// Die MoE-Architektur ermoeglicht effiziente Skalierung durch
// selektive Aktivierung von Experten pro Token.
package glm4moelite

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// MLP definiert die Schnittstelle fuer Feed-Forward-Netzwerke.
// Wird sowohl von sparse (MoE) als auch dense implementiert.
type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

// sparse implementiert ein Mixture-of-Experts (MoE) Netzwerk.
// Verwendet einen Router um Tokens an die besten Experten zu leiten.
type sparse struct {
	Router       *nn.Linear `gguf:"ffn_gate_inp"`  // Router fuer Experten-Auswahl
	Gate         *nn.Linear `gguf:"ffn_gate_exps"` // Gate-Projektion der Experten
	Up           *nn.Linear `gguf:"ffn_up_exps"`   // Up-Projektion der Experten
	Down         *nn.Linear `gguf:"ffn_down_exps"` // Down-Projektion der Experten
	SharedExpert *dense     `gguf:",suf:_shexp"`   // Gemeinsamer Experte fuer alle Tokens
	ExpProbsBias ml.Tensor  `gguf:"exp_probs_b.bias,alt:exp_probs_b"` // Bias fuer Experten-Wahrscheinlichkeiten
}

// Moe fuehrt die Mixture-of-Experts Berechnung durch.
// Verarbeitet hiddenStates mit den ausgewaehlten Experten basierend auf topKIndices.
func (moe *sparse) Moe(ctx ml.Context, hiddenStates, topKIndices, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	// Experten-Berechnungen mit Index-basierter Matrix-Multiplikation
	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = hiddenStates.SILU(ctx, upStates)

	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	experts = experts.Mul(ctx, topKWeights)

	// Experten-Ausgaben aggregieren
	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}
	return nextStates
}

// topKIndices berechnet die Indizes der Top-K Experten basierend auf Scores.
// Wendet optionalen Bias an bevor die besten Experten ausgewaehlt werden.
func (moe *sparse) topKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	if moe.ExpProbsBias != nil {
		scores = scores.Add(ctx, moe.ExpProbsBias)
	}
	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	return topKIndices
}

// Forward fuehrt den MoE Forward-Pass durch.
// Kombiniert Router-basierte Experten-Auswahl mit einem gemeinsamen Experten.
func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	residuals := hiddenStates

	// Router-Logits berechnen und Top-K Experten auswaehlen
	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	scores := routerLogits.Sigmoid(ctx)
	topKIndices := moe.topKIndices(ctx, scores, opts)
	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)

	// Optionale Normalisierung der Top-K Gewichte
	if opts.normTopKProb {
		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	// Skalierung und MoE-Berechnung
	topKWeights = topKWeights.Scale(ctx, float64(opts.routedScalingFactor))
	hiddenStates = moe.Moe(ctx, hiddenStates, topKIndices, topKWeights, opts)

	// Gemeinsamen Experten hinzufuegen
	sharedExpertResult := moe.SharedExpert.Forward(ctx, residuals, opts)
	hiddenStates = hiddenStates.Add(ctx, sharedExpertResult)
	return hiddenStates
}

// dense implementiert ein Standard Feed-Forward Netzwerk.
// Verwendet Gate-Up-Down Architektur mit SiLU Aktivierung.
type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"` // Gate-Projektion
	Up   *nn.Linear `gguf:"ffn_up"`   // Up-Projektion
	Down *nn.Linear `gguf:"ffn_down"` // Down-Projektion
}

// Forward fuehrt den Dense Forward-Pass durch.
// Berechnet: Down(SiLU(Gate(x)) * Up(x))
func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}
