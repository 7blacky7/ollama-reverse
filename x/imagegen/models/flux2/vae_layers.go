//go:build mlx

// vae_layers.go - Grundlegende Layer-Implementierungen für den VAE
// Enthält GroupNorm, Conv2D und ResNet-Block Definitionen

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// GroupNormLayer implements group normalization
// Reused from zimage package pattern
type GroupNormLayer struct {
	Weight    *mlx.Array `weight:"weight"`
	Bias      *mlx.Array `weight:"bias"`
	NumGroups int32
	Eps       float32
}

// Forward applies group normalization
// Input and output are in NHWC format [B, H, W, C]
func (gn *GroupNormLayer) Forward(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	// Reshape to [B, H, W, groups, C/groups]
	groupSize := C / gn.NumGroups
	x = mlx.Reshape(x, B, H, W, gn.NumGroups, groupSize)

	// Compute mean and variance per group
	mean := mlx.Mean(x, 1, true)
	mean = mlx.Mean(mean, 2, true)
	mean = mlx.Mean(mean, 4, true)

	xCentered := mlx.Sub(x, mean)

	sq := mlx.Square(xCentered)
	variance := mlx.Mean(sq, 1, true)
	variance = mlx.Mean(variance, 2, true)
	variance = mlx.Mean(variance, 4, true)

	// Normalize
	xNorm := mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, gn.Eps)))

	// Reshape back to [B, H, W, C]
	xNorm = mlx.Reshape(xNorm, B, H, W, C)

	// Scale and shift
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

// Conv2D represents a 2D convolution layer (reused pattern)
type Conv2D struct {
	Weight  *mlx.Array `weight:"weight"`
	Bias    *mlx.Array `weight:"bias,optional"`
	Stride  int32
	Padding int32
}

// Transform implements safetensors.Transformer to transpose weights from PyTorch's OIHW to MLX's OHWI.
func (conv *Conv2D) Transform(field string, arr *mlx.Array) *mlx.Array {
	if field == "Weight" {
		return mlx.Transpose(arr, 0, 2, 3, 1)
	}
	return arr
}

// Forward applies convolution (NHWC format)
func (conv *Conv2D) Forward(x *mlx.Array) *mlx.Array {
	out := mlx.Conv2d(x, conv.Weight, conv.Stride, conv.Padding)

	if conv.Bias != nil {
		bias := mlx.Reshape(conv.Bias, 1, 1, 1, conv.Bias.Dim(0))
		out = mlx.Add(out, bias)
	}

	return out
}
