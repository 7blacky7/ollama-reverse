//go:build mlx

// vae_blocks.go - ResNet, Attention und Mid-Block Implementierungen
// Enthält die Bausteine für Encoder und Decoder des VAE

package flux2

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// ResnetBlock2D implements a ResNet block for VAE
type ResnetBlock2D struct {
	Norm1        *GroupNormLayer `weight:"norm1"`
	Conv1        *Conv2D         `weight:"conv1"`
	Norm2        *GroupNormLayer `weight:"norm2"`
	Conv2        *Conv2D         `weight:"conv2"`
	ConvShortcut *Conv2D         `weight:"conv_shortcut,optional"`
}

// Forward applies the ResNet block
func (rb *ResnetBlock2D) Forward(x *mlx.Array) *mlx.Array {
	h := rb.Norm1.Forward(x)
	h = mlx.SiLU(h)
	h = rb.Conv1.Forward(h)

	h = rb.Norm2.Forward(h)
	h = mlx.SiLU(h)
	h = rb.Conv2.Forward(h)

	if rb.ConvShortcut != nil {
		x = rb.ConvShortcut.Forward(x)
	}

	return mlx.Add(h, x)
}

// VAEAttentionBlock implements self-attention for VAE
type VAEAttentionBlock struct {
	GroupNorm *GroupNormLayer `weight:"group_norm"`
	ToQ       nn.LinearLayer  `weight:"to_q"`
	ToK       nn.LinearLayer  `weight:"to_k"`
	ToV       nn.LinearLayer  `weight:"to_v"`
	ToOut     nn.LinearLayer  `weight:"to_out.0"`
}

// Forward applies attention (NHWC format)
func (ab *VAEAttentionBlock) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	shape := x.Shape()
	B := shape[0]
	H := shape[1]
	W := shape[2]
	C := shape[3]

	h := ab.GroupNorm.Forward(x)
	h = mlx.Reshape(h, B, H*W, C)

	q := ab.ToQ.Forward(h)
	k := ab.ToK.Forward(h)
	v := ab.ToV.Forward(h)

	q = mlx.ExpandDims(q, 1)
	k = mlx.ExpandDims(k, 1)
	v = mlx.ExpandDims(v, 1)

	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)
	out = mlx.Squeeze(out, 1)

	out = ab.ToOut.Forward(out)
	out = mlx.Reshape(out, B, H, W, C)
	out = mlx.Add(out, residual)

	return out
}

// VAEMidBlock is the middle block with attention
type VAEMidBlock struct {
	Resnet1   *ResnetBlock2D
	Attention *VAEAttentionBlock
	Resnet2   *ResnetBlock2D
}

// Forward applies the mid block
func (mb *VAEMidBlock) Forward(x *mlx.Array) *mlx.Array {
	x = mb.Resnet1.Forward(x)
	x = mb.Attention.Forward(x)
	x = mb.Resnet2.Forward(x)
	return x
}

// DownEncoderBlock2D implements a downsampling encoder block
type DownEncoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Downsample   *Conv2D
}

// Forward applies the down encoder block
func (db *DownEncoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range db.ResnetBlocks {
		x = resnet.Forward(x)
	}

	if db.Downsample != nil {
		// Pad then conv with stride 2
		x = mlx.Pad(x, []int32{0, 0, 0, 1, 0, 1, 0, 0})
		x = db.Downsample.Forward(x)
	}

	return x
}

// UpDecoderBlock2D implements an upsampling decoder block
type UpDecoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Upsample     *Conv2D
}

// Forward applies the up decoder block
func (ub *UpDecoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range ub.ResnetBlocks {
		x = resnet.Forward(x)
	}

	if ub.Upsample != nil {
		x = upsample2x(x)
		x = ub.Upsample.Forward(x)
	}

	return x
}

// upsample2x performs 2x nearest neighbor upsampling
func upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

	hIdx := mlx.ArangeInt(0, H, 1, mlx.DtypeInt32)
	hIdx = mlx.Reshape(hIdx, H, 1)
	hIdx = mlx.BroadcastTo(hIdx, []int32{H, 2})
	hIdx = mlx.Reshape(hIdx, H*2)

	wIdx := mlx.ArangeInt(0, W, 1, mlx.DtypeInt32)
	wIdx = mlx.Reshape(wIdx, W, 1)
	wIdx = mlx.BroadcastTo(wIdx, []int32{W, 2})
	wIdx = mlx.Reshape(wIdx, W*2)

	x = mlx.Take(x, hIdx, 1)
	x = mlx.Take(x, wIdx, 2)

	return x
}
