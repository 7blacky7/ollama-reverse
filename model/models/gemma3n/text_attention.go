// text_attention.go - Attention- und MLP-Komponenten für Gemma3n
// Enthält: TextAttention, TextMLP
package gemma3n

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type TextAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (attn TextAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, sharedKV bool, ropeBase float32, opts *TextOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	query := attn.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	query = attn.QueryNorm.Forward(ctx, query, opts.eps)
	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions, ropeBase)

	var key, value ml.Tensor
	if !sharedKV {
		key = attn.Key.Forward(ctx, hiddenStates)
		key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
		key = attn.KeyNorm.Forward(ctx, key, opts.eps)
		key = opts.applyRotaryPositionEmbeddings(ctx, key, positions, ropeBase)

		value = attn.Value.Forward(ctx, hiddenStates)
		value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
		value = value.RMSNorm(ctx, nil, opts.eps)
	}

	attention := nn.Attention(ctx, query, key, value, 1., cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return attn.Output.Forward(ctx, attention)
}

// TextMLP implementiert das Feed-Forward-Netzwerk mit optionaler Aktivierungssparsity
type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp TextMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, activationSparsityScale float64) ml.Tensor {
	upStates := mlp.Up.Forward(ctx, hiddenStates)
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates)
	if activationSparsityScale > 0 {
		mean := hiddenStates.Mean(ctx)
		std := hiddenStates.Stddev(ctx).Scale(ctx, activationSparsityScale)
		cutoff := mean.Add(ctx, std)
		hiddenStates = hiddenStates.Sub(ctx, cutoff).RELU(ctx)
	}

	hiddenStates = hiddenStates.GELU(ctx, upStates)
	hiddenStates = mlp.Down.Forward(ctx, hiddenStates)
	return hiddenStates
}
