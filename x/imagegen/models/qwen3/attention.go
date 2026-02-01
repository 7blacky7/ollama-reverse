//go:build mlx

// Modul: attention.go
// Beschreibung: Attention-Mechanismus für den Qwen3 Text-Encoder.
// Enthält: Attention-Struct, Forward-Methode, applyRoPEQwen3, repeatKV.

package qwen3

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Attention implements Qwen3 attention with QK norms
type Attention struct {
	QProj nn.LinearLayer `weight:"q_proj"`
	KProj nn.LinearLayer `weight:"k_proj"`
	VProj nn.LinearLayer `weight:"v_proj"`
	OProj nn.LinearLayer `weight:"o_proj"`
	QNorm *nn.RMSNorm    `weight:"q_norm"`
	KNorm *nn.RMSNorm    `weight:"k_norm"`
	// Computed fields
	NHeads    int32
	NKVHeads  int32
	HeadDim   int32
	Scale     float32
	RopeTheta float32
}

// applyRoPEQwen3 applies the custom RoPE for Qwen3 text encoder
func applyRoPEQwen3(x *mlx.Array, seqLen int32, theta float32) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	H := shape[2]
	D := shape[3]
	half := D / 2

	freqsArr := make([]float32, half)
	logTheta := float32(math.Log(float64(theta)))
	for i := int32(0); i < half; i++ {
		freqsArr[i] = float32(math.Exp(float64(-logTheta * float32(i) / float32(half))))
	}
	freqs := mlx.NewArray(freqsArr, []int32{half})

	posArr := make([]float32, seqLen)
	for i := int32(0); i < seqLen; i++ {
		posArr[i] = float32(i)
	}
	pos := mlx.NewArray(posArr, []int32{seqLen})

	posExpanded := mlx.Reshape(pos, seqLen, 1)
	freqsExpanded := mlx.Reshape(freqs, 1, half)
	args := mlx.Mul(posExpanded, freqsExpanded)

	cosVals := mlx.Cos(args)
	sinVals := mlx.Sin(args)
	cosVals = mlx.Reshape(cosVals, seqLen, 1, half)
	sinVals = mlx.Reshape(sinVals, seqLen, 1, half)

	x1 := mlx.Slice(x, []int32{0, 0, 0, 0}, []int32{B, L, H, half})
	x2 := mlx.Slice(x, []int32{0, 0, 0, half}, []int32{B, L, H, D})

	part1 := mlx.Sub(mlx.Mul(x1, cosVals), mlx.Mul(x2, sinVals))
	part2 := mlx.Add(mlx.Mul(x1, sinVals), mlx.Mul(x2, cosVals))

	return mlx.Concatenate([]*mlx.Array{part1, part2}, 3)
}

// Forward computes attention with causal masking and optional padding mask
func (attn *Attention) Forward(x *mlx.Array, mask *mlx.Array, maskMode string) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]

	q := attn.QProj.Forward(x)
	k := attn.KProj.Forward(x)
	v := attn.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, attn.NHeads, attn.HeadDim)
	k = mlx.Reshape(k, B, L, attn.NKVHeads, attn.HeadDim)
	v = mlx.Reshape(v, B, L, attn.NKVHeads, attn.HeadDim)

	// QK norm uses 1e-6 hardcoded (Qwen3 specific)
	q = attn.QNorm.Forward(q, 1e-6)
	k = attn.KNorm.Forward(k, 1e-6)

	q = applyRoPEQwen3(q, L, attn.RopeTheta)
	k = applyRoPEQwen3(k, L, attn.RopeTheta)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	if attn.NKVHeads < attn.NHeads {
		repeats := attn.NHeads / attn.NKVHeads
		k = repeatKV(k, repeats)
		v = repeatKV(v, repeats)
	}

	out := mlx.ScaledDotProductAttentionWithSinks(q, k, v, attn.Scale, maskMode, mask, nil)

	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, L, attn.NHeads*attn.HeadDim)

	out = attn.OProj.Forward(out)

	return out
}

// repeatKV repeats key/value heads for GQA
func repeatKV(x *mlx.Array, repeats int32) *mlx.Array {
	if repeats == 1 {
		return x
	}
	shape := x.Shape()
	x = mlx.ExpandDims(x, 2)
	x = mlx.Tile(x, []int32{1, 1, repeats, 1, 1})
	return mlx.Reshape(x, shape[0], shape[1]*repeats, shape[2], shape[3])
}
