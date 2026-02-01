//go:build mlx

// vae_decoder.go - VAE Decoder Hauptimplementierung
// Enthält den vollständigen VAEDecoder mit Load, Decode und Hilfsfunktionen.

package zimage

import (
	"fmt"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/vae"
)

// VAEDecoder is the full VAE decoder
type VAEDecoder struct {
	Config      *VAEConfig
	ConvIn      *Conv2D
	MidBlock    *VAEMidBlock
	UpBlocks    []*UpDecoderBlock2D
	ConvNormOut *GroupNormLayer
	ConvOut     *Conv2D

	// Tiling configuration (nil = no tiling)
	Tiling *vae.TilingConfig
}

// Load loads the VAE decoder from ollama blob storage.
func (m *VAEDecoder) Load(manifest *imagegen.ModelManifest) error {
	// Load config from blob
	var cfg VAEConfig
	if err := manifest.ReadConfigJSON("vae/config.json", &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	m.Config = &cfg

	// Load weights from tensor blobs
	weights, err := imagegen.LoadWeightsFromManifest(manifest, "vae")
	if err != nil {
		return fmt.Errorf("weights: %w", err)
	}
	if err := weights.Load(0); err != nil {
		return fmt.Errorf("load weights: %w", err)
	}
	defer weights.ReleaseAll()

	return m.loadWeights(weights, &cfg)
}

// loadWeights loads VAE weights from any WeightSource
func (m *VAEDecoder) loadWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load conv_in
	fmt.Print("  Loading conv_in... ")
	convInWeight, err := weights.GetTensor("decoder.conv_in.weight")
	if err != nil {
		return err
	}
	convInBias, err := weights.GetTensor("decoder.conv_in.bias")
	if err != nil {
		return err
	}
	m.ConvIn = NewConv2D(convInWeight, convInBias, 1, 1)
	fmt.Println("✓")

	// Load mid block
	fmt.Print("  Loading mid block... ")
	m.MidBlock, err = NewVAEMidBlock(weights, "decoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return err
	}
	fmt.Println("✓")

	// Load up blocks
	fmt.Print("  Loading up blocks... ")
	numBlocks := len(cfg.BlockOutChannels)
	m.UpBlocks = make([]*UpDecoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		hasUpsample := i < numBlocks-1
		m.UpBlocks[i], err = NewUpDecoderBlock2D(weights, prefix, cfg.LayersPerBlock+1, cfg.NormNumGroups, hasUpsample)
		if err != nil {
			return err
		}
	}
	fmt.Printf("✓ [%d blocks]\n", numBlocks)

	// Load conv_norm_out
	fmt.Print("  Loading conv_norm_out... ")
	normWeight, err := weights.GetTensor("decoder.conv_norm_out.weight")
	if err != nil {
		return err
	}
	normBias, err := weights.GetTensor("decoder.conv_norm_out.bias")
	if err != nil {
		return err
	}
	m.ConvNormOut = NewGroupNorm(normWeight, normBias, cfg.NormNumGroups)
	fmt.Println("✓")

	// Load conv_out
	fmt.Print("  Loading conv_out... ")
	convOutWeight, err := weights.GetTensor("decoder.conv_out.weight")
	if err != nil {
		return err
	}
	convOutBias, err := weights.GetTensor("decoder.conv_out.bias")
	if err != nil {
		return err
	}
	m.ConvOut = NewConv2D(convOutWeight, convOutBias, 1, 1)
	fmt.Println("✓")

	return nil
}

// Decode decodes latents to images.
// Input latents are in NCHW format, output is in NCHW format.
// If Tiling is set, uses tiled decoding to reduce memory for large images.
func (v *VAEDecoder) Decode(latents *mlx.Array) *mlx.Array {
	// Scale latents
	z := mlx.DivScalar(latents, v.Config.ScalingFactor)
	z = mlx.AddScalar(z, v.Config.ShiftFactor)
	// Convert NCHW -> NHWC for internal processing
	z = mlx.Transpose(z, 0, 2, 3, 1)

	// Use tiled decoding if enabled
	if v.Tiling != nil {
		mlx.Eval(z)
		return vae.DecodeTiled(z, v.Tiling, v.decodeTile)
	}

	// Direct decode
	h := v.decodeTile(z)
	h = mlx.ClipScalar(h, 0.0, 1.0, true, true)
	// Convert NHWC -> NCHW for output
	h = mlx.Transpose(h, 0, 3, 1, 2)
	mlx.Eval(h)
	return h
}

// decodeTile decodes a single latent tile to pixels.
// Input: [B, H, W, C] latent tile in NHWC format (already scaled)
// Output: [B, H*8, W*8, 3] pixel tile in NHWC format
func (v *VAEDecoder) decodeTile(z *mlx.Array) *mlx.Array {
	h := v.ConvIn.Forward(z)
	mlx.Eval(h)

	prev := h
	h = v.MidBlock.Forward(h)
	prev.Free()

	for _, upBlock := range v.UpBlocks {
		prev = h
		h = upBlock.Forward(h)
		prev.Free()
	}

	prev = h
	h = v.ConvNormOut.Forward(h)
	mlx.Eval(h) // Eval after GroupNorm to avoid grid dimension issues
	prev.Free()

	prev = h
	h = mlx.SiLU(h)
	h = v.ConvOut.Forward(h)
	mlx.Eval(h)
	prev.Free()

	// VAE outputs [-1, 1], convert to [0, 1]
	h = mlx.MulScalar(h, 0.5)
	h = mlx.AddScalar(h, 0.5)

	return h
}

// Upsample2x performs 2x nearest neighbor upsampling using Take.
// Input and output are in NHWC format: [B, H, W, C] -> [B, H*2, W*2, C]
// Uses Take with repeated indices to produce contiguous output.
func Upsample2x(x *mlx.Array) *mlx.Array {
	shape := x.Shape()
	H := shape[1]
	W := shape[2]

	// Create indices [0, 0, 1, 1, 2, 2, ...] for nearest neighbor
	// For H dimension
	hIdx := mlx.ArangeInt(0, H, 1, mlx.DtypeInt32)
	hIdx = mlx.Reshape(hIdx, H, 1)
	hIdx = mlx.BroadcastTo(hIdx, []int32{H, 2})
	hIdx = mlx.Reshape(hIdx, H*2)

	// For W dimension
	wIdx := mlx.ArangeInt(0, W, 1, mlx.DtypeInt32)
	wIdx = mlx.Reshape(wIdx, W, 1)
	wIdx = mlx.BroadcastTo(wIdx, []int32{W, 2})
	wIdx = mlx.Reshape(wIdx, W*2)

	// Take along H axis (axis 1 in NHWC)
	x = mlx.Take(x, hIdx, 1)
	// Take along W axis (axis 2 in NHWC)
	x = mlx.Take(x, wIdx, 2)

	return x
}
