//go:build mlx

// vae_layers.go - Grundlegende Layer-Implementierungen für den VAE Decoder
// Enthält GroupNorm und Conv2D Layer mit NHWC-Format Unterstützung.

package zimage

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// GroupNormLayer implements group normalization
type GroupNormLayer struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	NumGroups int32
	Eps       float32
}

// NewGroupNorm creates a group norm layer
func NewGroupNorm(weight, bias *mlx.Array, numGroups int32) *GroupNormLayer {
	return &GroupNormLayer{
		Weight:    weight,
		Bias:      bias,
		NumGroups: numGroups,
		Eps:       1e-5,
	}
}

// Forward applies group normalization
// Input and output are in NHWC format [B, H, W, C]
func (gn *GroupNormLayer) Forward(x *mlx.Array) *mlx.Array {
	// x: [B, H, W, C] (NHWC format)
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	// For large spatial sizes, use tiled computation to avoid CUDA grid limits
	// CUDA grid.y max is 65535, so H*W/16 must be <= 65535, meaning H*W <= ~1M
	// To be safe, tile when H*W > 512*512 = 262144
	if H*W > 512*512 {
		return gn.forwardTiled(x, B, H, W, C)
	}

	return gn.forwardSmall(x, B, H, W, C)
}

// forwardSmall is the standard GroupNorm for tensors that fit within CUDA grid limits
func (gn *GroupNormLayer) forwardSmall(x *mlx.Array, B, H, W, C int32) *mlx.Array {
	// Reshape to [B, H, W, groups, C/groups]
	groupSize := C / gn.NumGroups
	x = mlx.Reshape(x, B, H, W, gn.NumGroups, groupSize)

	// Compute mean and variance per group (over H, W, and C/groups dimensions)
	mean := mlx.Mean(x, 1, true)
	mean = mlx.Mean(mean, 2, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)

	// Variance over same axes
	sq := mlx.Square(xCentered)
	variance := mlx.Mean(sq, 1, true)
	variance = mlx.Mean(variance, 2, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back to [B, H, W, C]
	xNorm = mlx.Reshape(xNorm, B, H, W, C)

	// Scale and shift (weight and bias are [C])
	if gn.Weight != nil {
		weight := mlx.Reshape(gn.Weight, 1, 1, 1, C)
		xNorm = mlx.Mul(xNorm, weight)
	}
	if gn.Bias != nil {
		bias := mlx.Reshape(gn.Bias, 1, 1, 1, C)
		xNorm = mlx.Add(xNorm, bias)
	}

	return xNorm
}

// forwardTiled handles large tensors by processing in H-tiles to avoid CUDA grid limits
func (gn *GroupNormLayer) forwardTiled(x *mlx.Array, B, H, W, C int32) *mlx.Array {
	groupSize := C / gn.NumGroups

	// Keep the input - we need it for slicing tiles later
	// Track if we were the ones who kept it, so we can restore state after
	wasKept := x.Kept()
	mlx.Keep(x)

	// Compute per-group mean and variance using flattened spatial dimensions
	// Build the entire compute graph first, then eval once
	// Reshape to [B, H*W, groups, groupSize]
	xFlat := mlx.Reshape(x, B, H*W, gn.NumGroups, groupSize)

	// Mean over spatial (axis 1) and groupSize (axis 3) dimensions
	// Result shape: [B, 1, groups, 1]
	mean1 := mlx.Mean(xFlat, 1, true)
	mean := mlx.Mean(mean1, 3, true)

	// Variance using E[X^2] - E[X]^2
	xSq := mlx.Square(xFlat)
	meanSq1 := mlx.Mean(xSq, 1, true)
	meanSq := mlx.Mean(meanSq1, 3, true)
	meanSquared := mlx.Square(mean)
	variance := mlx.Sub(meanSq, meanSquared)

	// invStd = 1/sqrt(var + eps)
	varPlusEps := mlx.AddScalar(variance, gn.Eps)
	stdDev := mlx.Sqrt(varPlusEps)
	one := mlx.Full(1.0, 1)
	invStd := mlx.Div(one, stdDev)

	// Eval mean and invStd together - these are what we need for the tile loop
	mlx.Keep(mean, invStd)
	mlx.Eval(mean, invStd)

	// Tile along H dimension
	tileH := int32(512 * 512 / W)
	if tileH < 1 {
		tileH = 1
	}
	if tileH > H {
		tileH = H
	}

	// Prepare weight and bias reshaped for 4D broadcast [1, 1, groups, groupSize]
	var weightGN, biasGN *mlx.Array
	if gn.Weight != nil {
		weightGN = mlx.Reshape(gn.Weight, 1, 1, gn.NumGroups, groupSize)
		mlx.Keep(weightGN)
		mlx.Eval(weightGN)
	}
	if gn.Bias != nil {
		biasGN = mlx.Reshape(gn.Bias, 1, 1, gn.NumGroups, groupSize)
		mlx.Keep(biasGN)
		mlx.Eval(biasGN)
	}

	var tiles []*mlx.Array
	for hStart := int32(0); hStart < H; hStart += tileH {
		hEnd := hStart + tileH
		if hEnd > H {
			hEnd = H
		}
		tileHeight := hEnd - hStart
		spatialSize := tileHeight * W

		// Build the compute graph for this tile (no intermediate Evals)
		// Extract tile and flatten spatial dims: [B, tileH*W, groups, groupSize]
		tile := mlx.Slice(x, []int32{0, hStart, 0, 0}, []int32{B, hEnd, W, C})
		tileFlat := mlx.Reshape(tile, B, spatialSize, gn.NumGroups, groupSize)

		// Normalize: (x - mean) * invStd
		tileCentered := mlx.Sub(tileFlat, mean)
		tileNorm := mlx.Mul(tileCentered, invStd)

		// Apply scale and shift in 4D space
		if weightGN != nil {
			tileNorm = mlx.Mul(tileNorm, weightGN)
		}
		if biasGN != nil {
			tileNorm = mlx.Add(tileNorm, biasGN)
		}

		// Reshape back to [B, tileH, W, C]
		tileOut := mlx.Reshape(tileNorm, B, tileHeight, W, C)

		// Now eval and keep this tile
		mlx.Keep(tileOut)
		mlx.Eval(tileOut)

		tiles = append(tiles, tileOut)
	}

	// Concatenate tiles along H axis
	var result *mlx.Array
	if len(tiles) == 1 {
		result = tiles[0]
	} else {
		result = mlx.Concatenate(tiles, 1)
		mlx.Eval(result)
		// Free the individual tiles now that they're concatenated
		for _, t := range tiles {
			t.Free()
		}
	}

	// Clean up kept arrays
	// Restore x's kept state - only free if we were the ones who kept it
	if !wasKept {
		x.Free()
	}
	mean.Free()
	invStd.Free()
	if weightGN != nil {
		weightGN.Free()
	}
	if biasGN != nil {
		biasGN.Free()
	}

	return result
}

// Conv2D represents a 2D convolution layer
// Works natively in NHWC format (MLX's native format)
type Conv2D struct {
	Weight  *mlx.Array // [out_channels, kH, kW, in_channels] (OHWI for MLX)
	Bias    *mlx.Array // [out_channels]
	Stride  int32
	Padding int32
}

// NewConv2D creates a Conv2D layer
// weight comes in as [out_channels, in_channels, kH, kW] (OIHW from PyTorch)
// we transpose to [out_channels, kH, kW, in_channels] (OHWI for MLX)
func NewConv2D(weight, bias *mlx.Array, stride, padding int32) *Conv2D {
	// Transpose weight from OIHW to OHWI
	// [O, I, H, W] -> [O, H, W, I]
	weightOHWI := mlx.Transpose(weight, 0, 2, 3, 1)
	return &Conv2D{
		Weight:  weightOHWI,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}
}

// Forward applies convolution
// Input and output are in NHWC format [N, H, W, C]
func (conv *Conv2D) Forward(x *mlx.Array) *mlx.Array {
	// Conv in NHWC format (MLX native)
	out := mlx.Conv2d(x, conv.Weight, conv.Stride, conv.Padding)

	if conv.Bias != nil {
		// Bias is [C], reshape to [1, 1, 1, C] for NHWC broadcast
		bias := mlx.Reshape(conv.Bias, 1, 1, 1, conv.Bias.Dim(0))
		out = mlx.Add(out, bias)
	}

	return out
}
