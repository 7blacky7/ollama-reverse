//go:build mlx

// vae_attention.go - VAE Attention Block Implementierung
// Enthält den VAEAttentionBlock für Self-Attention im VAE Decoder.

package zimage

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// VAEAttentionBlock implements self-attention for VAE
type VAEAttentionBlock struct {
	GroupNorm   *GroupNormLayer
	ToQWeight   *mlx.Array
	ToQBias     *mlx.Array
	ToKWeight   *mlx.Array
	ToKBias     *mlx.Array
	ToVWeight   *mlx.Array
	ToVBias     *mlx.Array
	ToOutWeight *mlx.Array
	ToOutBias   *mlx.Array
	NumHeads    int32
}

// NewVAEAttentionBlock creates an attention block
func NewVAEAttentionBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
	normWeight, err := weights.GetTensor(prefix + ".group_norm.weight")
	if err != nil {
		return nil, err
	}
	normBias, err := weights.GetTensor(prefix + ".group_norm.bias")
	if err != nil {
		return nil, err
	}

	toQWeight, err := weights.GetTensor(prefix + ".to_q.weight")
	if err != nil {
		return nil, err
	}
	toQBias, err := weights.GetTensor(prefix + ".to_q.bias")
	if err != nil {
		return nil, err
	}

	toKWeight, err := weights.GetTensor(prefix + ".to_k.weight")
	if err != nil {
		return nil, err
	}
	toKBias, err := weights.GetTensor(prefix + ".to_k.bias")
	if err != nil {
		return nil, err
	}

	toVWeight, err := weights.GetTensor(prefix + ".to_v.weight")
	if err != nil {
		return nil, err
	}
	toVBias, err := weights.GetTensor(prefix + ".to_v.bias")
	if err != nil {
		return nil, err
	}

	toOutWeight, err := weights.GetTensor(prefix + ".to_out.0.weight")
	if err != nil {
		return nil, err
	}
	toOutBias, err := weights.GetTensor(prefix + ".to_out.0.bias")
	if err != nil {
		return nil, err
	}

	return &VAEAttentionBlock{
		GroupNorm:   NewGroupNorm(normWeight, normBias, numGroups),
		ToQWeight:   mlx.Transpose(toQWeight, 1, 0),
		ToQBias:     toQBias,
		ToKWeight:   mlx.Transpose(toKWeight, 1, 0),
		ToKBias:     toKBias,
		ToVWeight:   mlx.Transpose(toVWeight, 1, 0),
		ToVBias:     toVBias,
		ToOutWeight: mlx.Transpose(toOutWeight, 1, 0),
		ToOutBias:   toOutBias,
		NumHeads:    1,
	}, nil
}

// Forward applies attention with staged evaluation
// Input and output are in NHWC format [B, H, W, C]
func (ab *VAEAttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	var h *mlx.Array

	// Stage 1: GroupNorm + reshape to [B, H*W, C]
	{
		h = ab.GroupNorm.Forward(x)
		h = mlx.Reshape(h, B, H*W, C)
		mlx.Eval(h)
	}

	var out *mlx.Array

	// Stage 2: Q, K, V projections + attention
	{
		q := mlx.Linear(h, ab.ToQWeight)
		q = mlx.Add(q, ab.ToQBias)
		k := mlx.Linear(h, ab.ToKWeight)
		k = mlx.Add(k, ab.ToKBias)
		v := mlx.Linear(h, ab.ToVWeight)
		v = mlx.Add(v, ab.ToVBias)
		h.Free()

		q = mlx.ExpandDims(q, 1)
		k = mlx.ExpandDims(k, 1)
		v = mlx.ExpandDims(v, 1)

		scale := float32(1.0 / math.Sqrt(float64(C)))
		out = mlx.ScaledDotProductAttention(q, k, v, scale, false)
		out = mlx.Squeeze(out, 1)
		mlx.Eval(out)
	}

	// Stage 3: Output projection + reshape + residual
	{
		prev := out
		out = mlx.Linear(out, ab.ToOutWeight)
		out = mlx.Add(out, ab.ToOutBias)
		out = mlx.Reshape(out, B, H, W, C)
		out = mlx.Add(out, residual)
		prev.Free()
		mlx.Eval(out)
	}

	return out
}
