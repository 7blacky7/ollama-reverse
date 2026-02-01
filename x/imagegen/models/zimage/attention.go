//go:build mlx

// Package zimage - Attention-Implementierung
// Enthält: Attention mit QKV-Fusion und RoPE-Anwendung

package zimage

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Attention implements multi-head attention with QK norm
type Attention struct {
	ToQ   nn.LinearLayer `weight:"to_q"`
	ToK   nn.LinearLayer `weight:"to_k"`
	ToV   nn.LinearLayer `weight:"to_v"`
	ToOut nn.LinearLayer `weight:"to_out.0"`
	NormQ *mlx.Array     `weight:"norm_q.weight"` // [head_dim] for per-head RMSNorm
	NormK *mlx.Array     `weight:"norm_k.weight"`
	// Fused QKV (computed at init time for efficiency, not loaded from weights)
	ToQKV nn.LinearLayer `weight:"-"` // Fused Q+K+V projection (created by FuseQKV)
	Fused bool           `weight:"-"` // Whether to use fused QKV path
	// Computed fields (not loaded from weights)
	NHeads  int32   `weight:"-"`
	HeadDim int32   `weight:"-"`
	Dim     int32   `weight:"-"`
	Scale   float32 `weight:"-"`
}

// FuseQKV creates a fused QKV projection by concatenating weights.
// This reduces 3 matmuls to 1 for a ~5-10% speedup.
// Note: Fusion is skipped for quantized weights as it would require complex
// dequant-concat-requant operations. The FP8 memory bandwidth savings outweigh
// the ~5% fusion benefit.
func (attn *Attention) FuseQKV() {
	if attn.ToQ == nil || attn.ToK == nil || attn.ToV == nil {
		return
	}

	// Skip fusion for quantized weights - type assert to check
	toQ, qOk := attn.ToQ.(*nn.Linear)
	toK, kOk := attn.ToK.(*nn.Linear)
	toV, vOk := attn.ToV.(*nn.Linear)
	if !qOk || !kOk || !vOk {
		// One or more are QuantizedLinear, skip fusion
		return
	}

	if toQ.Weight == nil || toK.Weight == nil || toV.Weight == nil {
		return
	}

	// Concatenate weights: [dim, dim] x 3 -> [3*dim, dim]
	// Weight shapes: ToQ.Weight [out_dim, in_dim], etc.
	qWeight := toQ.Weight
	kWeight := toK.Weight
	vWeight := toV.Weight

	// Concatenate along output dimension (axis 0)
	fusedWeight := mlx.Concatenate([]*mlx.Array{qWeight, kWeight, vWeight}, 0)

	// Evaluate fused weight to ensure it's materialized
	mlx.Eval(fusedWeight)

	// Create fused linear layer
	fusedLinear := &nn.Linear{Weight: fusedWeight}

	// Handle bias if present
	if toQ.Bias != nil && toK.Bias != nil && toV.Bias != nil {
		fusedBias := mlx.Concatenate([]*mlx.Array{toQ.Bias, toK.Bias, toV.Bias}, 0)
		mlx.Eval(fusedBias)
		fusedLinear.Bias = fusedBias
	}

	attn.ToQKV = fusedLinear
	attn.Fused = true
}

// Forward computes attention
func (attn *Attention) Forward(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]

	xFlat := mlx.Reshape(x, B*L, D)

	var q, k, v *mlx.Array
	if attn.Fused && attn.ToQKV != nil {
		// Fused QKV path: single matmul then split
		qkv := attn.ToQKV.Forward(xFlat) // [B*L, 3*dim]

		// Split into Q, K, V along last dimension
		// Each has shape [B*L, dim]
		q = mlx.Slice(qkv, []int32{0, 0}, []int32{B * L, attn.Dim})
		k = mlx.Slice(qkv, []int32{0, attn.Dim}, []int32{B * L, 2 * attn.Dim})
		v = mlx.Slice(qkv, []int32{0, 2 * attn.Dim}, []int32{B * L, 3 * attn.Dim})
	} else {
		// Separate Q, K, V projections
		q = attn.ToQ.Forward(xFlat)
		k = attn.ToK.Forward(xFlat)
		v = attn.ToV.Forward(xFlat)
	}

	// Reshape to [B, L, nheads, head_dim]
	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NHeads, attn.HeadDim)

	// QK norm
	q = mlx.RMSNorm(q, attn.NormQ, 1e-5)
	k = mlx.RMSNorm(k, attn.NormK, 1e-5)

	// Apply RoPE if provided
	if cos != nil && sin != nil {
		q = applyRoPE3D(q, cos, sin)
		k = applyRoPE3D(k, cos, sin)
	}

	// Transpose to [B, nheads, L, head_dim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// SDPA
	out := mlx.ScaledDotProductAttention(q, k, v, attn.Scale, false)

	// Transpose back and reshape
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B*L, attn.Dim)
	out = attn.ToOut.Forward(out)

	return mlx.Reshape(out, B, L, attn.Dim)
}

// applyRoPE3D applies 3-axis rotary position embeddings
// x: [B, L, nheads, head_dim]
// cos, sin: [B, L, 1, head_dim/2]
func applyRoPE3D(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	nheads := shape[2]
	headDim := shape[3]
	half := headDim / 2

	// Create even/odd index arrays
	evenIdx := make([]int32, half)
	oddIdx := make([]int32, half)
	for i := int32(0); i < half; i++ {
		evenIdx[i] = i * 2
		oddIdx[i] = i*2 + 1
	}
	evenIndices := mlx.NewArrayInt32(evenIdx, []int32{half})
	oddIndices := mlx.NewArrayInt32(oddIdx, []int32{half})

	// Extract x1 (even indices) and x2 (odd indices) along last axis
	x1 := mlx.Take(x, evenIndices, 3) // [B, L, nheads, half]
	x2 := mlx.Take(x, oddIndices, 3)  // [B, L, nheads, half]

	// Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
	r1 := mlx.Sub(mlx.Mul(x1, cos), mlx.Mul(x2, sin))
	r2 := mlx.Add(mlx.Mul(x1, sin), mlx.Mul(x2, cos))

	// Stack and reshape to interleave: [r1_0, r2_0, r1_1, r2_1, ...]
	r1 = mlx.ExpandDims(r1, 4)                          // [B, L, nheads, half, 1]
	r2 = mlx.ExpandDims(r2, 4)                          // [B, L, nheads, half, 1]
	stacked := mlx.Concatenate([]*mlx.Array{r1, r2}, 4) // [B, L, nheads, half, 2]
	return mlx.Reshape(stacked, B, L, nheads, headDim)
}

// initTransformerBlock sets computed fields on a transformer block
func initTransformerBlock(block *TransformerBlock, cfg *TransformerConfig) {
	block.Dim = cfg.Dim
	block.HasModulation = block.AdaLN != nil

	// Init attention computed fields
	attn := block.Attention
	attn.NHeads = cfg.NHeads
	attn.HeadDim = cfg.Dim / cfg.NHeads
	attn.Dim = cfg.Dim
	attn.Scale = float32(1.0 / math.Sqrt(float64(attn.HeadDim)))

	// Init feedforward OutDim
	block.FeedForward.OutDim = block.FeedForward.W2.OutputDim()

	// Set eps on all RMSNorm layers
	block.AttentionNorm1.Eps = cfg.NormEps
	block.AttentionNorm2.Eps = cfg.NormEps
	block.FFNNorm1.Eps = cfg.NormEps
	block.FFNNorm2.Eps = cfg.NormEps
}
