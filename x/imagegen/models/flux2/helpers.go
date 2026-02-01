//go:build mlx

// Package flux2 - Hilfsfunktionen
// Enthält: parseModulation3, modulateLayerNorm, splitQKV, applyQKNorm, compiledSwiGLU

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// parseModulation3 extracts 3 modulation params (shift, scale, gate) starting at offset
func parseModulation3(mod *mlx.Array, dim int32, offset int32) (*mlx.Array, *mlx.Array, *mlx.Array) {
	B := mod.Shape()[0]
	start := offset * dim
	shift := mlx.Slice(mod, []int32{0, start}, []int32{B, start + dim})
	scale := mlx.Slice(mod, []int32{0, start + dim}, []int32{B, start + 2*dim})
	gate := mlx.Slice(mod, []int32{0, start + 2*dim}, []int32{B, start + 3*dim})

	// Expand for broadcasting [B, dim] -> [B, 1, dim]
	shift = mlx.ExpandDims(shift, 1)
	scale = mlx.ExpandDims(scale, 1)
	gate = mlx.ExpandDims(gate, 1)

	return shift, scale, gate
}

// modulateLayerNorm applies LayerNorm then shift/scale modulation
// Diffusers uses LayerNorm(elementwise_affine=False) which centers the data
func modulateLayerNorm(x *mlx.Array, shift, scale *mlx.Array) *mlx.Array {
	// Fast LayerNorm without learnable params
	x = mlx.LayerNorm(x, 1e-6)

	// Modulate: x * (1 + scale) + shift
	x = mlx.Mul(x, mlx.AddScalar(scale, 1.0))
	return mlx.Add(x, shift)
}

// splitQKV splits a fused QKV tensor into Q, K, V
func splitQKV(qkv *mlx.Array, B, L, dim int32) (*mlx.Array, *mlx.Array, *mlx.Array) {
	q := mlx.Slice(qkv, []int32{0, 0, 0}, []int32{B, L, dim})
	k := mlx.Slice(qkv, []int32{0, 0, dim}, []int32{B, L, 2 * dim})
	v := mlx.Slice(qkv, []int32{0, 0, 2 * dim}, []int32{B, L, 3 * dim})
	return q, k, v
}

// applyQKNorm applies RMSNorm with learned scale (no bias)
// Uses the optimized mlx_fast_rms_norm
func applyQKNorm(x *mlx.Array, scale *mlx.Array) *mlx.Array {
	return mlx.RMSNorm(x, scale, 1e-6)
}

// compiledSwiGLU fuses: silu(gate) * up
// Called 30x per step (10 in dual-stream + 20 in single-stream blocks)
var compiledSwiGLU *mlx.CompiledFunc

func getCompiledSwiGLU() *mlx.CompiledFunc {
	if compiledSwiGLU == nil {
		compiledSwiGLU = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			gate, up := inputs[0], inputs[1]
			return []*mlx.Array{mlx.Mul(mlx.SiLU(gate), up)}
		}, true)
	}
	return compiledSwiGLU
}
