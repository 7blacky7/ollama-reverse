// model_text.go - Hauptmodul für das Gemma3n Text-Modell
// Enthält: TextModel, Forward-Methode, TextScaledWordEmbedding, PerLayerProjector
package gemma3n

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model/input"
)

type TextModel struct {
	TokenEmbedding *TextScaledWordEmbedding `gguf:"token_embd"`

	*PerLayerProjector

	AltupEmbd   *nn.Linear `gguf:"altup_proj"`
	AltupUnembd *nn.Linear `gguf:"altup_unembd_proj"`

	TextLayers []TextLayer `gguf:"blk"`
	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	TextOptions
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	// Create a tensor of a single float32 value of 1.0 to use for altup correction
	one := ctx.Input().FromFloats([]float32{1.0}, 1)

	inputs := m.TokenEmbedding.Forward(ctx, batch.Inputs, math.Sqrt(float64(m.hiddenSize)))
	inputsPerLayer := m.PerLayerProjector.Forward(ctx, batch, inputs, &m.TextOptions)

	targetMagnitude := inputs.Sqr(ctx).Mean(ctx).Sqrt(ctx)
	targetMagnitude = targetMagnitude.Repeat(ctx, 2, m.altupInputs-1)

	hiddenState := inputs.Repeat(ctx, 2, m.altupInputs-1)
	altupProj := m.AltupEmbd.Forward(ctx, hiddenState)
	altupProj = altupProj.Mul(ctx, targetMagnitude.Div(ctx, altupProj.Sqr(ctx).Mean(ctx).Sqrt(ctx)))

	hiddenStates := inputs.Concat(ctx, altupProj, 2)

	firstSharedKeyValue := m.hiddenLayers - m.sharedKeyValueLayers
	for i, layer := range m.TextLayers {
		if i < firstSharedKeyValue {
			cache.SetLayer(i)
		} else if m.isLocal(i) {
			cache.SetLayer(firstSharedKeyValue - 2)
		} else {
			cache.SetLayer(firstSharedKeyValue - 1)
		}

		var layerType int
		ropeBase := m.ropeBase
		if m.isLocal(i) {
			layerType = 1
			ropeBase = m.ropeBaseLocal
		}

		cache.(*kvcache.WrapperCache).SetLayerType(layerType)

		// inputPerLayer = inputsPerLayer[:, i, :].squeeze(1)
		inputPerLayer := inputsPerLayer.View(ctx, i*inputsPerLayer.Stride(1), inputsPerLayer.Dim(0), inputsPerLayer.Stride(2), inputsPerLayer.Dim(2))
		hiddenStates = layer.Forward(ctx, hiddenStates, inputPerLayer, positions, one, cache, i >= firstSharedKeyValue, ropeBase, float64(m.activationSparsityScale[i]), &m.TextOptions)
	}

	// hiddenStates = hiddenStates[:, :, 0]
	hiddenStates0 := hiddenStates.Slice(ctx, 2, 0, 1, 1)
	targetMagnitude = hiddenStates0.Sqr(ctx).Mean(ctx).Sqrt(ctx)
	targetMagnitude = targetMagnitude.Repeat(ctx, 2, m.altupInputs-1)

	// hiddenState = hiddenStates[:, :, 1:]
	hiddenState = hiddenStates.Slice(ctx, 2, 1, hiddenStates.Dim(2), 1)
	altupUnembdProj := m.AltupUnembd.Forward(ctx, hiddenState)
	altupUnembdProj = altupUnembdProj.Mul(ctx, targetMagnitude.Div(ctx, altupUnembdProj.Sqr(ctx).Mean(ctx).Sqrt(ctx)))

	hiddenStates = hiddenStates0.Concat(ctx, altupUnembdProj, 2)

	hiddenStates = hiddenStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx).Mean(ctx)
	hiddenStates = hiddenStates.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	hiddenStates = hiddenStates.Rows(ctx, batch.Outputs)

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase := m.ropeBase
	if m.isLocal(layer) {
		ropeBase = m.ropeBaseLocal
	}

	return m.applyRotaryPositionEmbeddings(ctx, key, shift, ropeBase), nil
}

type TextScaledWordEmbedding struct {
	*nn.Embedding
}

func (e TextScaledWordEmbedding) Forward(ctx ml.Context, inputIDs ml.Tensor, scale float64) ml.Tensor {
	return e.Embedding.Forward(ctx, inputIDs).Scale(ctx, scale)
}

type PerLayerProjector struct {
	TokenEmbedding *TextScaledWordEmbedding `gguf:"per_layer_token_embd"`
	Projector      *nn.Linear               `gguf:"per_layer_model_proj"`
	Norm           *nn.RMSNorm              `gguf:"per_layer_proj_norm"`
}

func (p PerLayerProjector) Forward(ctx ml.Context, batch input.Batch, inputs ml.Tensor, opts *TextOptions) ml.Tensor {
	inputsPerLayer := p.TokenEmbedding.Forward(ctx, batch.Inputs, math.Sqrt(float64(opts.hiddenSizePerLayerInput)))
	inputsPerLayer = inputsPerLayer.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, batch.Inputs.Dim(0), batch.Inputs.Dim(1))

	perLayerProjection := p.Projector.Forward(ctx, inputs)
	perLayerProjection = perLayerProjection.Scale(ctx, math.Sqrt(float64(opts.hiddenSize)))
	perLayerProjection = perLayerProjection.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, inputs.Dim(1))
	perLayerProjection = p.Norm.Forward(ctx, perLayerProjection, opts.eps)

	if inputsPerLayer != nil {
		perLayerProjection = perLayerProjection.Add(ctx, inputsPerLayer)
		perLayerProjection = perLayerProjection.Scale(ctx, 1/math.Sqrt(2))
	}

	return perLayerProjection
}

// newTextModel erstellt ein neues TextModel mit den gegebenen Konfigurationsoptionen
func newTextModel(c fs.Config) *TextModel {
	return &TextModel{
		TextLayers: make([]TextLayer, c.Uint("block_count")),
		TextOptions: TextOptions{
			hiddenLayers:            int(c.Uint("block_count")),
			hiddenSize:              int(c.Uint("embedding_length")),
			hiddenSizePerLayerInput: int(c.Uint("embedding_length_per_layer_input")),
			numHeads:                int(c.Uint("attention.head_count")),
			numKVHeads:              int(c.Uint("attention.head_count_kv")),
			keyLength:               int(c.Uint("attention.key_length")),
			valueLength:             int(c.Uint("attention.value_length")),
			sharedKeyValueLayers:    int(c.Uint("attention.shared_kv_layers")),

			altupActiveIndex: int(c.Uint("altup.active_idx")),
			altupInputs:      int(c.Uint("altup.num_inputs")),

			eps:           c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeBase:      c.Float("rope.freq_base", 1_000_000),
			ropeBaseLocal: c.Float("rope.freq_base_local", 10_000),
			ropeScale:     c.Float("rope.scaling.factor", 1.0),

			slidingWindowPattern:    c.Bools("attention.sliding_window_pattern"),
			activationSparsityScale: c.Floats("activation_sparsity_scale"),
		},
	}
}
