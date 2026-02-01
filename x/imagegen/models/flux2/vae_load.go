//go:build mlx

// vae_load.go - Weight-Loading-Funktionen fuer den Flux2 VAE
// Enthält alle Funktionen zum Laden der Modellgewichte aus Safetensors

package flux2

import (
	"fmt"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// Load loads the Flux2 VAE from ollama blob storage.
func (m *AutoencoderKLFlux2) Load(manifest *imagegen.ModelManifest) error {
	fmt.Print("  Loading VAE... ")

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
func (m *AutoencoderKLFlux2) loadWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load encoder components (for image conditioning)
	if err := m.loadEncoderWeights(weights, cfg); err != nil {
		return fmt.Errorf("encoder: %w", err)
	}

	// Load decoder conv_in
	m.DecoderConvIn = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.DecoderConvIn, weights, "decoder.conv_in"); err != nil {
		return fmt.Errorf("decoder.conv_in: %w", err)
	}

	// Load mid block
	m.DecoderMid, err = loadVAEMidBlock(weights, "decoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return fmt.Errorf("decoder.mid_block: %w", err)
	}

	// Load up blocks
	numBlocks := len(cfg.BlockOutChannels)
	m.DecoderUp = make([]*UpDecoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		hasUpsample := i < numBlocks-1
		m.DecoderUp[i], err = loadUpDecoderBlock2D(weights, prefix, cfg.LayersPerBlock+1, cfg.NormNumGroups, hasUpsample)
		if err != nil {
			return fmt.Errorf("%s: %w", prefix, err)
		}
	}

	// Load decoder conv_norm_out and conv_out
	m.DecoderNormOut = &GroupNormLayer{NumGroups: cfg.NormNumGroups, Eps: 1e-5}
	if err := safetensors.LoadModule(m.DecoderNormOut, weights, "decoder.conv_norm_out"); err != nil {
		return fmt.Errorf("decoder.conv_norm_out: %w", err)
	}

	m.DecoderConvOut = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.DecoderConvOut, weights, "decoder.conv_out"); err != nil {
		return fmt.Errorf("decoder.conv_out: %w", err)
	}

	// Load post_quant_conv
	if cfg.UsePostQuantConv {
		m.PostQuantConv = &Conv2D{Stride: 1, Padding: 0}
		if err := safetensors.LoadModule(m.PostQuantConv, weights, "post_quant_conv"); err != nil {
			return fmt.Errorf("post_quant_conv: %w", err)
		}
	}

	// Load latent BatchNorm (affine=False, so no weight/bias)
	bnMean, err := weights.GetTensor("bn.running_mean")
	if err != nil {
		return fmt.Errorf("bn.running_mean: %w", err)
	}
	bnVar, err := weights.GetTensor("bn.running_var")
	if err != nil {
		return fmt.Errorf("bn.running_var: %w", err)
	}
	m.LatentBN = &BatchNorm2D{
		RunningMean: bnMean,
		RunningVar:  bnVar,
		Weight:      nil, // affine=False
		Bias:        nil, // affine=False
		Eps:         cfg.BatchNormEps,
		Momentum:    cfg.BatchNormMomentum,
	}

	fmt.Println("\u2713")
	return nil
}

// loadVAEMidBlock loads the mid block.
func loadVAEMidBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEMidBlock, error) {
	resnet1, err := loadResnetBlock2D(weights, prefix+".resnets.0", numGroups)
	if err != nil {
		return nil, err
	}

	attention, err := loadVAEAttentionBlock(weights, prefix+".attentions.0", numGroups)
	if err != nil {
		return nil, err
	}

	resnet2, err := loadResnetBlock2D(weights, prefix+".resnets.1", numGroups)
	if err != nil {
		return nil, err
	}

	return &VAEMidBlock{
		Resnet1:   resnet1,
		Attention: attention,
		Resnet2:   resnet2,
	}, nil
}

// loadResnetBlock2D loads a ResNet block.
func loadResnetBlock2D(weights safetensors.WeightSource, prefix string, numGroups int32) (*ResnetBlock2D, error) {
	block := &ResnetBlock2D{
		Norm1:        &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
		Conv1:        &Conv2D{Stride: 1, Padding: 1},
		Norm2:        &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
		Conv2:        &Conv2D{Stride: 1, Padding: 1},
		ConvShortcut: &Conv2D{Stride: 1, Padding: 0}, // Pre-allocate for optional loading
	}
	if err := safetensors.LoadModule(block, weights, prefix); err != nil {
		return nil, err
	}
	// If ConvShortcut wasn't loaded (no weights found), nil it out
	if block.ConvShortcut.Weight == nil {
		block.ConvShortcut = nil
	}
	return block, nil
}

// loadVAEAttentionBlock loads an attention block using LoadModule.
func loadVAEAttentionBlock(weights safetensors.WeightSource, prefix string, numGroups int32) (*VAEAttentionBlock, error) {
	ab := &VAEAttentionBlock{
		GroupNorm: &GroupNormLayer{NumGroups: numGroups, Eps: 1e-5},
	}
	if err := safetensors.LoadModule(ab, weights, prefix); err != nil {
		return nil, err
	}
	return ab, nil
}

// loadUpDecoderBlock2D loads an up decoder block.
func loadUpDecoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasUpsample bool) (*UpDecoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := loadResnetBlock2D(weights, resPrefix, numGroups)
		if err != nil {
			return nil, err
		}
		resnets[i] = resnet
	}

	var upsample *Conv2D
	if hasUpsample {
		upsample = &Conv2D{Stride: 1, Padding: 1}
		if err := safetensors.LoadModule(upsample, weights, prefix+".upsamplers.0.conv"); err != nil {
			return nil, err
		}
	}

	return &UpDecoderBlock2D{
		ResnetBlocks: resnets,
		Upsample:     upsample,
	}, nil
}

// loadEncoderWeights loads the encoder components for image conditioning
func (m *AutoencoderKLFlux2) loadEncoderWeights(weights safetensors.WeightSource, cfg *VAEConfig) error {
	var err error

	// Load encoder conv_in
	m.EncoderConvIn = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.EncoderConvIn, weights, "encoder.conv_in"); err != nil {
		return fmt.Errorf("encoder.conv_in: %w", err)
	}

	// Load encoder down blocks
	numBlocks := len(cfg.BlockOutChannels)
	m.EncoderDown = make([]*DownEncoderBlock2D, numBlocks)
	for i := 0; i < numBlocks; i++ {
		prefix := fmt.Sprintf("encoder.down_blocks.%d", i)
		hasDownsample := i < numBlocks-1
		m.EncoderDown[i], err = loadDownEncoderBlock2D(weights, prefix, cfg.LayersPerBlock, cfg.NormNumGroups, hasDownsample)
		if err != nil {
			return fmt.Errorf("%s: %w", prefix, err)
		}
	}

	// Load encoder mid block
	m.EncoderMid, err = loadVAEMidBlock(weights, "encoder.mid_block", cfg.NormNumGroups)
	if err != nil {
		return fmt.Errorf("encoder.mid_block: %w", err)
	}

	// Load encoder conv_norm_out and conv_out
	m.EncoderNormOut = &GroupNormLayer{NumGroups: cfg.NormNumGroups, Eps: 1e-5}
	if err := safetensors.LoadModule(m.EncoderNormOut, weights, "encoder.conv_norm_out"); err != nil {
		return fmt.Errorf("encoder.conv_norm_out: %w", err)
	}

	m.EncoderConvOut = &Conv2D{Stride: 1, Padding: 1}
	if err := safetensors.LoadModule(m.EncoderConvOut, weights, "encoder.conv_out"); err != nil {
		return fmt.Errorf("encoder.conv_out: %w", err)
	}

	// Load quant_conv (for encoding)
	if cfg.UseQuantConv {
		m.QuantConv = &Conv2D{Stride: 1, Padding: 0}
		if err := safetensors.LoadModule(m.QuantConv, weights, "quant_conv"); err != nil {
			return fmt.Errorf("quant_conv: %w", err)
		}
	}

	return nil
}

// loadDownEncoderBlock2D loads a down encoder block.
func loadDownEncoderBlock2D(weights safetensors.WeightSource, prefix string, numLayers, numGroups int32, hasDownsample bool) (*DownEncoderBlock2D, error) {
	resnets := make([]*ResnetBlock2D, numLayers)
	for i := int32(0); i < numLayers; i++ {
		resPrefix := fmt.Sprintf("%s.resnets.%d", prefix, i)
		resnet, err := loadResnetBlock2D(weights, resPrefix, numGroups)
		if err != nil {
			return nil, err
		}
		resnets[i] = resnet
	}

	var downsample *Conv2D
	if hasDownsample {
		downsample = &Conv2D{Stride: 2, Padding: 0}
		if err := safetensors.LoadModule(downsample, weights, prefix+".downsamplers.0.conv"); err != nil {
			return nil, err
		}
	}

	return &DownEncoderBlock2D{
		ResnetBlocks: resnets,
		Downsample:   downsample,
	}, nil
}
