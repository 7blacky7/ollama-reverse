//go:build mlx

// Package zimage - RoPE (Rotary Position Embedding) Implementierung
// Enthält: RoPECache, prepareRoPE3D, createCoordinateGrid

package zimage

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// RoPECache holds precomputed RoPE values
type RoPECache struct {
	ImgCos     *mlx.Array
	ImgSin     *mlx.Array
	CapCos     *mlx.Array
	CapSin     *mlx.Array
	UnifiedCos *mlx.Array
	UnifiedSin *mlx.Array
	ImgLen     int32
	CapLen     int32
	GridH      int32 // Image token grid height
	GridW      int32 // Image token grid width
}

// PrepareRoPECache precomputes RoPE values for the given image and caption lengths.
// hTok and wTok are the number of tokens in each dimension (latentH/patchSize, latentW/patchSize).
func (m *Transformer) PrepareRoPECache(hTok, wTok, capLen int32) *RoPECache {
	imgLen := hTok * wTok

	// Image positions: grid over (1, H, W) starting at (capLen+1, 0, 0)
	imgPos := createCoordinateGrid(1, hTok, wTok, capLen+1, 0, 0)
	imgPos = mlx.ToBFloat16(imgPos)
	// Caption positions: grid over (capLen, 1, 1) starting at (1, 0, 0)
	capPos := createCoordinateGrid(capLen, 1, 1, 1, 0, 0)
	capPos = mlx.ToBFloat16(capPos)

	// Compute RoPE from UNIFIED positions
	unifiedPos := mlx.Concatenate([]*mlx.Array{imgPos, capPos}, 1)
	unifiedCos, unifiedSin := prepareRoPE3D(unifiedPos, m.TransformerConfig.AxesDims)

	// Slice RoPE for image and caption parts
	imgCos := mlx.Slice(unifiedCos, []int32{0, 0, 0, 0}, []int32{1, imgLen, 1, 64})
	imgSin := mlx.Slice(unifiedSin, []int32{0, 0, 0, 0}, []int32{1, imgLen, 1, 64})
	capCos := mlx.Slice(unifiedCos, []int32{0, imgLen, 0, 0}, []int32{1, imgLen + capLen, 1, 64})
	capSin := mlx.Slice(unifiedSin, []int32{0, imgLen, 0, 0}, []int32{1, imgLen + capLen, 1, 64})

	return &RoPECache{
		ImgCos:     imgCos,
		ImgSin:     imgSin,
		CapCos:     capCos,
		CapSin:     capSin,
		UnifiedCos: unifiedCos,
		UnifiedSin: unifiedSin,
		ImgLen:     imgLen,
		CapLen:     capLen,
		GridH:      hTok,
		GridW:      wTok,
	}
}

// createCoordinateGrid creates 3D position grid [1, d0*d1*d2, 3]
func createCoordinateGrid(d0, d1, d2, s0, s1, s2 int32) *mlx.Array {
	// Create meshgrid and stack
	total := d0 * d1 * d2
	coords := make([]float32, total*3)

	idx := 0
	for i := int32(0); i < d0; i++ {
		for j := int32(0); j < d1; j++ {
			for k := int32(0); k < d2; k++ {
				coords[idx*3+0] = float32(s0 + i)
				coords[idx*3+1] = float32(s1 + j)
				coords[idx*3+2] = float32(s2 + k)
				idx++
			}
		}
	}

	return mlx.NewArray(coords, []int32{1, total, 3})
}

// prepareRoPE3D computes cos/sin for 3-axis RoPE
// positions: [B, L, 3] with (h, w, t) coordinates
// axesDims: [32, 48, 48] - dimensions for each axis
// Returns: cos, sin each [B, L, 1, head_dim/2]
func prepareRoPE3D(positions *mlx.Array, axesDims []int32) (*mlx.Array, *mlx.Array) {
	// Compute frequencies for each axis
	// dims = [32, 48, 48], so halves = [16, 24, 24]
	ropeTheta := float32(256.0)

	freqs := make([]*mlx.Array, 3)
	for axis := 0; axis < 3; axis++ {
		half := axesDims[axis] / 2
		f := make([]float32, half)
		for i := int32(0); i < half; i++ {
			f[i] = float32(math.Exp(-math.Log(float64(ropeTheta)) * float64(i) / float64(half)))
		}
		freqs[axis] = mlx.NewArray(f, []int32{1, 1, 1, half})
	}

	// Extract position coordinates
	shape := positions.Shape()
	B := shape[0]
	L := shape[1]

	// positions[:, :, 0] -> h positions
	posH := mlx.Slice(positions, []int32{0, 0, 0}, []int32{B, L, 1})
	posW := mlx.Slice(positions, []int32{0, 0, 1}, []int32{B, L, 2})
	posT := mlx.Slice(positions, []int32{0, 0, 2}, []int32{B, L, 3})

	// Compute args: pos * freqs for each axis
	posH = mlx.ExpandDims(posH, 3) // [B, L, 1, 1]
	posW = mlx.ExpandDims(posW, 3)
	posT = mlx.ExpandDims(posT, 3)

	argsH := mlx.Mul(posH, freqs[0]) // [B, L, 1, 16]
	argsW := mlx.Mul(posW, freqs[1]) // [B, L, 1, 24]
	argsT := mlx.Mul(posT, freqs[2]) // [B, L, 1, 24]

	// Concatenate: [B, L, 1, 16+24+24=64]
	args := mlx.Concatenate([]*mlx.Array{argsH, argsW, argsT}, 3)

	// Compute cos and sin
	return mlx.Cos(args), mlx.Sin(args)
}
