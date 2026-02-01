//go:build mlx

// attention.go - Attention und MLP Implementierung fuer Gemma 3.
//
// Dieses Modul enthaelt:
// - Attention Forward mit Q/K Normalisierung und RoPE
// - MLP Forward mit GELU-Approximation (tanh-Variante)
// - Kompilierte GELU-Funktion fuer Performance

package gemma3

import (
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Forward runs attention with Q/K normalization
func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, isSliding bool, cfg *TextConfig) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim]
	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	// Q/K normalization after reshaping (use precomputed scaled weight)
	q = mlx.RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	k = mlx.RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)

	// Apply RoPE with appropriate theta
	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}
	q = mlx.RoPE(q, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	k = mlx.RoPE(k, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())

	// Update cache
	k, v = c.Update(k, v, int(L))

	// Repeat K/V for GQA if needed
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = nn.RepeatKV(k, repeatFactor)
		v = nn.RepeatKV(v, repeatFactor)
	}

	// Attention
	out := mlx.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// compiledGeluApprox is a singleton compiled GELU function shared across all layers
var compiledGeluApprox *mlx.CompiledFunc

// getCompiledGeluApprox returns the compiled GELU function, creating it once if needed
func getCompiledGeluApprox() *mlx.CompiledFunc {
	if compiledGeluApprox == nil {
		compiledGeluApprox = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			return []*mlx.Array{geluApproxImpl(inputs[0])}
		}, true)
	}
	return compiledGeluApprox
}

// Forward runs the MLP with GELU approximation (tanh variant)
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := getCompiledGeluApprox().Call(m.GateProj.Forward(x))[0]
	return m.DownProj.Forward(mlx.Mul(gate, m.UpProj.Forward(x)))
}

// geluApproxImpl computes GELU using the tanh approximation (gelu_pytorch_tanh):
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluApproxImpl(x *mlx.Array) *mlx.Array {
	// Constants
	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	const coeff = 0.044715

	// x^3
	x3 := mlx.Mul(mlx.Mul(x, x), x)
	// x + 0.044715 * x^3
	inner := mlx.Add(x, mlx.MulScalar(x3, coeff))
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled := mlx.MulScalar(inner, sqrt2OverPi)
	// tanh(...)
	tanh := mlx.Tanh(scaled)
	// 1 + tanh(...)
	onePlusTanh := mlx.AddScalar(tanh, 1.0)
	// 0.5 * x * (1 + tanh(...))
	return mlx.Mul(mlx.MulScalar(x, 0.5), onePlusTanh)
}

// gemmaRMSNorm applies Gemma-style RMS normalization: x * rsqrt(mean(x^2) + eps) * (1 + weight)
// Uses mlx.RMSNorm fast kernel with pre-computed (1 + weight)
func gemmaRMSNorm(x, weight *mlx.Array, eps float32) *mlx.Array {
	// Gemma uses (1 + weight) instead of weight
	scaledWeight := mlx.AddScalar(weight, 1.0)
	return mlx.RMSNorm(x, scaledWeight, eps)
}
