//go:build mlx

// Package zimage - Transformer-Bloecke
// Enthält: TransformerBlock mit AdaLN-Modulation und FinalLayer

package zimage

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// TransformerBlock is a single transformer block with optional AdaLN modulation
type TransformerBlock struct {
	Attention      *Attention     `weight:"attention"`
	FeedForward    *FeedForward   `weight:"feed_forward"`
	AttentionNorm1 *nn.RMSNorm    `weight:"attention_norm1"`
	AttentionNorm2 *nn.RMSNorm    `weight:"attention_norm2"`
	FFNNorm1       *nn.RMSNorm    `weight:"ffn_norm1"`
	FFNNorm2       *nn.RMSNorm    `weight:"ffn_norm2"`
	AdaLN          nn.LinearLayer `weight:"adaLN_modulation.0,optional"` // only if modulation
	// Computed fields
	HasModulation bool
	Dim           int32
}

// Forward applies the transformer block
func (tb *TransformerBlock) Forward(x *mlx.Array, adaln *mlx.Array, cos, sin *mlx.Array, eps float32) *mlx.Array {
	if tb.AdaLN != nil && adaln != nil {
		// Compute modulation: [B, 256] -> [B, 4*dim]
		chunks := tb.AdaLN.Forward(adaln)

		// Split into 4 parts: scale_msa, gate_msa, scale_mlp, gate_mlp
		chunkShape := chunks.Shape()
		chunkDim := chunkShape[1] / 4

		scaleMSA := mlx.Slice(chunks, []int32{0, 0}, []int32{chunkShape[0], chunkDim})
		gateMSA := mlx.Slice(chunks, []int32{0, chunkDim}, []int32{chunkShape[0], chunkDim * 2})
		scaleMLP := mlx.Slice(chunks, []int32{0, chunkDim * 2}, []int32{chunkShape[0], chunkDim * 3})
		gateMLP := mlx.Slice(chunks, []int32{0, chunkDim * 3}, []int32{chunkShape[0], chunkDim * 4})

		// Expand for broadcasting: [B, 1, dim]
		scaleMSA = mlx.ExpandDims(scaleMSA, 1)
		gateMSA = mlx.ExpandDims(gateMSA, 1)
		scaleMLP = mlx.ExpandDims(scaleMLP, 1)
		gateMLP = mlx.ExpandDims(gateMLP, 1)

		// Attention with modulation
		normX := tb.AttentionNorm1.Forward(x, eps)
		normX = mlx.Mul(normX, mlx.AddScalar(scaleMSA, 1.0))
		attnOut := tb.Attention.Forward(normX, cos, sin)
		attnOut = tb.AttentionNorm2.Forward(attnOut, eps)
		x = mlx.Add(x, mlx.Mul(mlx.Tanh(gateMSA), attnOut))

		// FFN with modulation
		normFFN := tb.FFNNorm1.Forward(x, eps)
		normFFN = mlx.Mul(normFFN, mlx.AddScalar(scaleMLP, 1.0))
		ffnOut := tb.FeedForward.Forward(normFFN)
		ffnOut = tb.FFNNorm2.Forward(ffnOut, eps)
		x = mlx.Add(x, mlx.Mul(mlx.Tanh(gateMLP), ffnOut))
	} else {
		// No modulation (context refiner)
		attnOut := tb.Attention.Forward(tb.AttentionNorm1.Forward(x, eps), cos, sin)
		x = mlx.Add(x, tb.AttentionNorm2.Forward(attnOut, eps))

		ffnOut := tb.FeedForward.Forward(tb.FFNNorm1.Forward(x, eps))
		x = mlx.Add(x, tb.FFNNorm2.Forward(ffnOut, eps))
	}

	return x
}

// FinalLayer outputs the denoised patches
type FinalLayer struct {
	AdaLN  nn.LinearLayer `weight:"adaLN_modulation.1"` // [256] -> [dim]
	Output nn.LinearLayer `weight:"linear"`             // [dim] -> [out_channels]
	OutDim int32          // computed from Output
}

// Forward computes final output
func (fl *FinalLayer) Forward(x *mlx.Array, c *mlx.Array) *mlx.Array {
	// c: [B, 256] -> scale: [B, dim]
	scale := mlx.SiLU(c)
	scale = fl.AdaLN.Forward(scale)
	scale = mlx.ExpandDims(scale, 1) // [B, 1, dim]

	// LayerNorm (affine=False) then scale
	x = layerNormNoAffine(x, 1e-6)
	x = mlx.Mul(x, mlx.AddScalar(scale, 1.0))

	// Output projection
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	D := shape[2]
	x = mlx.Reshape(x, B*L, D)
	x = fl.Output.Forward(x)

	return mlx.Reshape(x, B, L, fl.OutDim)
}

// layerNormNoAffine applies layer norm without learnable parameters
func layerNormNoAffine(x *mlx.Array, eps float32) *mlx.Array {
	ndim := x.Ndim()
	lastAxis := ndim - 1

	mean := mlx.Mean(x, lastAxis, true)
	xCentered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Square(xCentered), lastAxis, true)
	return mlx.Div(xCentered, mlx.Sqrt(mlx.AddScalar(variance, eps)))
}
