// text_layer.go - Transformer-Layer-Komponenten für Gemma3n
// Enthält: TextLayer, AltUp, Laurel
package gemma3n

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type TextLayer struct {
	*AltUp
	*Laurel

	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	Attention         *TextAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`

	MLPNorm     *nn.RMSNorm `gguf:"ffn_norm"`
	MLP         *TextMLP
	PostMLPNorm *nn.RMSNorm `gguf:"post_ffw_norm"`

	PerLayerInputGate  *nn.Linear  `gguf:"inp_gate"`
	PerLayerProjection *nn.Linear  `gguf:"proj"`
	PostPerLayerNorm   *nn.RMSNorm `gguf:"post_norm"`
}

func (d TextLayer) Forward(ctx ml.Context, hiddenStates, perLayerInput, positions, one ml.Tensor, cache kvcache.Cache, sharedKV bool, ropeBase float32, activationSparsityScale float64, opts *TextOptions) ml.Tensor {
	predictions := d.Predict(ctx, hiddenStates, opts)
	active := opts.altupActive(ctx, predictions)

	attn := d.AttentionNorm.Forward(ctx, active, opts.eps)
	laurel := d.Laurel.Forward(ctx, attn, opts)

	attn = d.Attention.Forward(ctx, attn, positions, cache, sharedKV, ropeBase, opts)
	attn = d.PostAttentionNorm.Forward(ctx, attn, opts.eps)
	attn = active.Add(ctx, attn)
	attn = attn.Add(ctx, laurel).Scale(ctx, 1/math.Sqrt(2))

	mlp := d.MLPNorm.Forward(ctx, attn, opts.eps)
	mlp = d.MLP.Forward(ctx, mlp, activationSparsityScale)
	mlp = d.PostMLPNorm.Forward(ctx, mlp, opts.eps)
	mlp = attn.Add(ctx, mlp)

	predictions = d.Correct(ctx, predictions, mlp, one, opts)
	active = opts.altupActive(ctx, predictions)
	if opts.altupCorrectScale {
		active = d.ScaleCorrectedOutput(ctx, active)
	}

	active = d.PerLayerInputGate.Forward(ctx, active)
	active = active.GELU(ctx, perLayerInput)

	active = d.PerLayerProjection.Forward(ctx, active)
	active = d.PostPerLayerNorm.Forward(ctx, active, opts.eps)

	// inactive := predictions[:, :, 1:]
	inactive := predictions.Slice(ctx, 2, 1, predictions.Dim(2), 1)
	active = inactive.Add(ctx, active)

	predictions0 := predictions.Slice(ctx, 2, 0, 1, 1)
	return predictions0.Concat(ctx, active, 2)
}

// AltUp implementiert den AltUp-Mechanismus für Gemma3n
// Verantwortlich für Prediction, Correction und Skalierung
type AltUp struct {
	CorrectionScale       ml.Tensor   `gguf:"altup_correct_scale.weight"`
	PredictionCoefficient *nn.Linear  `gguf:"altup_predict_coef"`
	CorrectionCoefficient *nn.Linear  `gguf:"altup_correct_coef"`
	Router                *nn.Linear  `gguf:"altup_router"`
	RouterNorm            *nn.RMSNorm `gguf:"altup_router_norm"`
}

func (a AltUp) computeRouterModalities(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	routerInputs := a.RouterNorm.Forward(ctx, hiddenStates, opts.eps).Scale(ctx, 1.0/float64(opts.hiddenSize))
	return a.Router.Forward(ctx, routerInputs).Tanh(ctx)
}

func (a AltUp) Predict(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	modalities := a.computeRouterModalities(ctx, opts.altupActive(ctx, hiddenStates), opts)

	coefficients := a.PredictionCoefficient.Forward(ctx, modalities)
	coefficients = coefficients.Reshape(ctx, opts.altupInputs, opts.altupInputs, coefficients.Dim(1), coefficients.Dim(2))

	predictions := coefficients.Mulmat(ctx, hiddenStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx))
	predictions = predictions.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	return predictions.Add(ctx, hiddenStates)
}

func (a AltUp) Correct(ctx ml.Context, predictions, activated, one ml.Tensor, opts *TextOptions) ml.Tensor {
	innovation := activated.Sub(ctx, opts.altupActive(ctx, predictions))
	innovation = innovation.Repeat(ctx, 2, opts.altupInputs)

	modalities := a.computeRouterModalities(ctx, activated, opts)
	coefficients := a.CorrectionCoefficient.Forward(ctx, modalities)
	coefficients = coefficients.Add(ctx, one)

	coefficients = coefficients.Reshape(ctx, 1, coefficients.Dim(0), coefficients.Dim(1))
	coefficients = coefficients.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	corrected := innovation.Mul(ctx, coefficients)
	corrected = corrected.Add(ctx, predictions)
	return corrected
}

func (a AltUp) ScaleCorrectedOutput(ctx ml.Context, predictions ml.Tensor) ml.Tensor {
	return predictions.Mul(ctx, a.CorrectionScale)
}

// Laurel implementiert die Low-Rank-Attention-Alternative
type Laurel struct {
	LinearLeft     *nn.Linear  `gguf:"laurel_l"`
	LinearRight    *nn.Linear  `gguf:"laurel_r"`
	PostLaurelNorm *nn.RMSNorm `gguf:"laurel_post_norm"`
}

func (l Laurel) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = l.LinearLeft.Forward(ctx, hiddenStates)
	hiddenStates = l.LinearRight.Forward(ctx, hiddenStates)
	hiddenStates = l.PostLaurelNorm.Forward(ctx, hiddenStates, opts.eps)
	return hiddenStates.Add(ctx, residual)
}
