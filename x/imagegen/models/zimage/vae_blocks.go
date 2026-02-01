//go:build mlx

// vae_blocks.go - VAE Decoder Blöcke
// Enthält ResnetBlock2D, UpDecoderBlock2D und VAEMidBlock.

package zimage

import (
	"fmt"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// ResnetBlock2D implements a ResNet block for VAE
type ResnetBlock2D struct {
	Norm1        *GroupNormLayer
	Conv1        *Conv2D
	Norm2        *GroupNormLayer
	Conv2        *Conv2D
	ConvShortcut *Conv2D // nil if in_channels == out_channels
}

// NewResnetBlock2D creates a ResNet block
func NewResnetBlock2D(weights safetensors.WeightSource, prefix string, numGroups int32) (*ResnetBlock2D, error) {
	norm1Weight, err := weights.GetTensor(prefix + ".norm1.weight")
	if err != nil {
		return nil, err
	}
	norm1Bias, err := weights.GetTensor(prefix + ".norm1.bias")
	if err != nil {
		return nil, err
	}

	conv1Weight, err := weights.GetTensor(prefix + ".conv1.weight")
	if err != nil {
		return nil, err
	}
	conv1Bias, err := weights.GetTensor(prefix + ".conv1.bias")
	if err != nil {
		return nil, err
	}

	norm2Weight, err := weights.GetTensor(prefix + ".norm2.weight")
	if err != nil {
		return nil, err
	}
	norm2Bias, err := weights.GetTensor(prefix + ".norm2.bias")
	if err != nil {
		return nil, err
	}

	conv2Weight, err := weights.GetTensor(prefix + ".conv2.weight")
	if err != nil {
		return nil, err
	}
	conv2Bias, err := weights.GetTensor(prefix + ".conv2.bias")
	if err != nil {
		return nil, err
	}

	block := &ResnetBlock2D{
		Norm1: NewGroupNorm(norm1Weight, norm1Bias, numGroups),
		Conv1: NewConv2D(conv1Weight, conv1Bias, 1, 1),
		Norm2: NewGroupNorm(norm2Weight, norm2Bias, numGroups),
		Conv2: NewConv2D(conv2Weight, conv2Bias, 1, 1),
	}

	if weights.HasTensor(prefix + ".conv_shortcut.weight") {
		shortcutWeight, err := weights.GetTensor(prefix + ".conv_shortcut.weight")
		if err != nil {
			return nil, err
		}
		shortcutBias, err := weights.GetTensor(prefix + ".conv_shortcut.bias")
		if err != nil {
			return nil, err
		}
		block.ConvShortcut = NewConv2D(shortcutWeight, shortcutBias, 1, 0)
	}

	return block, nil
}

// Forward applies the ResNet block with staged evaluation
func (rb *ResnetBlock2D) Forward(x *mlx.Array) *mlx.Array {
	var h *mlx.Array

	// Stage 1: norm1
	{
		h = rb.Norm1.Forward(x)
		mlx.Eval(h)
	}

	// Stage 2: silu + conv1
	{
		prev := h
		h = mlx.SiLU(h)
		h = rb.Conv1.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Stage 3: norm2
	{
		prev := h
		h = rb.Norm2.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Stage 4: silu + conv2
	{
		prev := h
		h = mlx.SiLU(h)
		h = rb.Conv2.Forward(h)
		prev.Free()
		mlx.Eval(h)
	}

	// Residual connection
	{
		prev := h
		if rb.ConvShortcut != nil {
			shortcut := rb.ConvShortcut.Forward(x)
			h = mlx.Add(h, shortcut)
		} else {
			h = mlx.Add(h, x)
		}
		prev.Free()
		mlx.Eval(h)
	}

	return h
}

// UpDecoderBlock2D implements an upsampling decoder block
type UpDecoderBlock2D struct {
	ResnetBlocks []*ResnetBlock2D
	Upsample     *Conv2D
}

// NewUpDecoderBlock2D creates an up decoder block
func NewUpDecoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasUpsample bool) (*UpDecoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := NewResnetBlock2D(weights, resPrefix, numGroups)
		if err != nil {
			return nil, err
		}
		resnets[i] = resnet
	}

	var upsample *Conv2D
	if hasUpsample {
		upWeight, err := weights.GetTensor(prefix + ".upsamplers.0.conv.weight")
		if err != nil {
			return nil, err
		}
		upBias, err := weights.GetTensor(prefix + ".upsamplers.0.conv.bias")
		if err != nil {
			return nil, err
		}
		upsample = NewConv2D(upWeight, upBias, 1, 1)
	}

	return &UpDecoderBlock2D{
		ResnetBlocks: resnets,
		Upsample:     upsample,
	}, nil
}

// Forward applies the up decoder block with staged evaluation to reduce peak memory
func (ub *UpDecoderBlock2D) Forward(x *mlx.Array) *mlx.Array {
	for _, resnet := range ub.ResnetBlocks {
		prev := x
		x = resnet.Forward(x) // ResNet handles its own pools
		prev.Free()
	}

	if ub.Upsample != nil {
		// Stage 1: Upsample2x (nearest neighbor)
		{
			prev := x
			x = Upsample2x(x)
			prev.Free()
			mlx.Eval(x)
		}

		// Stage 2: Upsample conv
		{
			prev := x
			x = ub.Upsample.Forward(x)
			prev.Free()
			mlx.Eval(x)
		}
	}

	return x
}

// VAEMidBlock is the middle block with attention
type VAEMidBlock struct {
	Resnet1   *ResnetBlock2D
	Attention *VAEAttentionBlock
	Resnet2   *ResnetBlock2D
}

// NewVAEMidBlock creates the mid block
func NewVAEMidBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEMidBlock, error) {
	resnet1, err := NewResnetBlock2D(weights, prefix+".resnets.0", numGroups)
	if err != nil {
		return nil, err
	}

	attention, err := NewVAEAttentionBlock(weights, prefix+".attentions.0", numGroups)
	if err != nil {
		return nil, err
	}

	resnet2, err := NewResnetBlock2D(weights, prefix+".resnets.1", numGroups)
	if err != nil {
		return nil, err
	}

	return &VAEMidBlock{
		Resnet1:   resnet1,
		Attention: attention,
		Resnet2:   resnet2,
	}, nil
}

// Forward applies the mid block with staged evaluation
func (mb *VAEMidBlock) Forward(x *mlx.Array) *mlx.Array {
	prev := x
	x = mb.Resnet1.Forward(x) // ResNet handles its own pools
	prev.Free()

	// Attention handles its own pools
	prev = x
	x = mb.Attention.Forward(x)
	prev.Free()

	prev = x
	x = mb.Resnet2.Forward(x) // ResNet handles its own pools
	prev.Free()

	return x
}
