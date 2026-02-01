package qwen25vl

import (
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// ============================================================================
// Vision Layers - Attention, MLP und Encoder-Layer fuer das Vision-Modell
// ============================================================================
//
// Dieses Modul enthaelt:
// - VisionSelfAttention: Self-Attention mit RoPE
// - VisionMLP: Gated MLP mit SILU-Aktivierung
// - VisionEncoderLayer: Kombiniert Attention und MLP mit Residual-Verbindungen

// VisionSelfAttention implementiert Self-Attention fuer das Vision-Modell
type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

// Forward fuehrt die Self-Attention Berechnung durch
func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1))
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1))
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1))

	// Rotary Position Embeddings anwenden
	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	// Skalierungsfaktor fuer Scaled Dot-Product Attention
	scale := 1.0 / math.Sqrt(float64(opts.headDim))

	// Scaled Dot-Product Attention berechnen
	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)
	kqv := value.Mulmat(ctx, kq)
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))

	return sa.Output.Forward(ctx, attention)
}

// VisionMLP implementiert das Multi-Layer Perceptron mit Gated Activation
type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

// Forward fuehrt die MLP-Berechnung durch (Gate * SILU(Up))
func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

// VisionEncoderLayer kombiniert Attention und MLP zu einem Transformer-Block
type VisionEncoderLayer struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

// Forward fuehrt einen Encoder-Layer durch (Attention -> Add -> MLP -> Add)
func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Attention mit Residual-Verbindung
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positions, mask, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	// MLP mit Residual-Verbindung
	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}
