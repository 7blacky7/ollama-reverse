//go:build mlx

// Package zimage - Patchify/Unpatchify Operationen
// Enthält: Funktionen zum Umwandeln von Latents in Patches und zurueck

package zimage

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// PatchifyLatents converts latents [B, C, H, W] to patches [B, L, C*patch^2]
// Matches Python: x.reshape(C, 1, 1, H_tok, 2, W_tok, 2).transpose(1,2,3,5,4,6,0).reshape(1,-1,C*4)
func PatchifyLatents(latents *mlx.Array, patchSize int32) *mlx.Array {
	shape := latents.Shape()
	C := shape[1]
	H := shape[2]
	W := shape[3]

	pH := H / patchSize // H_tok
	pW := W / patchSize // W_tok

	// Match Python exactly: reshape treating B=1 as part of contiguous data
	// [1, C, H, W] -> [C, 1, 1, pH, 2, pW, 2]
	x := mlx.Reshape(latents, C, 1, 1, pH, patchSize, pW, patchSize)

	// Python: transpose(1, 2, 3, 5, 4, 6, 0)
	// [C, 1, 1, pH, 2, pW, 2] -> [1, 1, pH, pW, 2, 2, C]
	x = mlx.Transpose(x, 1, 2, 3, 5, 4, 6, 0)

	// [1, 1, pH, pW, 2, 2, C] -> [1, pH*pW, C*4]
	return mlx.Reshape(x, 1, pH*pW, C*patchSize*patchSize)
}

// UnpatchifyLatents converts patches [B, L, C*patch^2] back to [B, C, H, W]
// Matches Python: out.reshape(1,1,H_tok,W_tok,2,2,C).transpose(6,0,1,2,4,3,5).reshape(1,C,H,W)
func UnpatchifyLatents(patches *mlx.Array, patchSize, H, W, C int32) *mlx.Array {
	pH := H / patchSize
	pW := W / patchSize

	// [1, L, C*4] -> [1, 1, pH, pW, 2, 2, C]
	x := mlx.Reshape(patches, 1, 1, pH, pW, patchSize, patchSize, C)

	// Python: transpose(6, 0, 1, 2, 4, 3, 5)
	// [1, 1, pH, pW, 2, 2, C] -> [C, 1, 1, pH, 2, pW, 2]
	x = mlx.Transpose(x, 6, 0, 1, 2, 4, 3, 5)

	// [C, 1, 1, pH, 2, pW, 2] -> [1, C, H, W]
	return mlx.Reshape(x, 1, C, H, W)
}
