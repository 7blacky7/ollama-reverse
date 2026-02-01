//go:build mlx

// utils.go - Hilfsfunktionen fuer Z-Image
// Enthaelt Utility-Funktionen wie Padding und Shift-Berechnung.
package zimage

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// padToLength pads a sequence tensor to the target length by repeating the last token.
func padToLength(x *mlx.Array, targetLen int32) *mlx.Array {
	shape := x.Shape()
	currentLen := shape[1]
	if currentLen >= targetLen {
		return x
	}
	padLen := targetLen - currentLen
	lastToken := mlx.Slice(x, []int32{0, currentLen - 1, 0}, []int32{shape[0], currentLen, shape[2]})
	padding := mlx.Tile(lastToken, []int32{1, padLen, 1})
	return mlx.Concatenate([]*mlx.Array{x, padding}, 1)
}

// CalculateShift computes the mu shift value for dynamic scheduling
func CalculateShift(imgSeqLen int32) float32 {
	baseSeqLen := float32(256)
	maxSeqLen := float32(4096)
	baseShift := float32(0.5)
	maxShift := float32(1.15)

	m := (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
	b := baseShift - m*baseSeqLen
	return float32(imgSeqLen)*m + b
}
