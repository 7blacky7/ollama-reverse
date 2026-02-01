//go:build mlx

// attention.go - Attention-Mechanismus fuer GPT-OSS
// Enthaelt Attention-Struktur, YaRN RoPE und Sliding Window Mask.
package gpt_oss

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Attention represents the attention layer
type Attention struct {
	QProj      *nn.Linear `weight:"self_attn.q_proj"`
	KProj      *nn.Linear `weight:"self_attn.k_proj"`
	VProj      *nn.Linear `weight:"self_attn.v_proj"`
	OProj      *nn.Linear `weight:"self_attn.o_proj"`
	Sinks      *mlx.Array `weight:"self_attn.sinks,optional"`
	YarnFreqs  *mlx.Array // computed
	YarnMscale float32
}

// ComputeYarnFreqs computes YaRN-modified RoPE frequencies
// Based on mlx-lm's YarnRoPE implementation
func ComputeYarnFreqs(dims int32, base, scalingFactor float32, origMaxPos int32, betaFast, betaSlow float32) (*mlx.Array, float32) {
	// yarn_find_correction_dim
	yarnFindCorrectionDim := func(numRotations float64) float64 {
		return float64(dims) * math.Log(float64(origMaxPos)/(numRotations*2*math.Pi)) / (2 * math.Log(float64(base)))
	}

	// yarn_find_correction_range
	low := int(math.Floor(yarnFindCorrectionDim(float64(betaFast))))
	high := int(math.Ceil(yarnFindCorrectionDim(float64(betaSlow))))
	if low < 0 {
		low = 0
	}
	if high > int(dims)-1 {
		high = int(dims) - 1
	}

	// yarn_get_mscale
	yarnGetMscale := func(scale, mscale float64) float64 {
		if scale <= 1 {
			return 1.0
		}
		return 0.1*mscale*math.Log(scale) + 1.0
	}
	mscale := float32(yarnGetMscale(float64(scalingFactor), 1.0) / yarnGetMscale(float64(scalingFactor), 0.0))

	// Compute frequencies
	halfDims := dims / 2
	freqData := make([]float32, halfDims)
	for i := int32(0); i < halfDims; i++ {
		exp := float64(2*i) / float64(dims)
		freqExtra := math.Pow(float64(base), exp)
		freqInter := float64(scalingFactor) * freqExtra

		// linear ramp mask
		var freqMask float64
		if low == high {
			freqMask = 0.0
		} else {
			t := (float64(i) - float64(low)) / float64(high-low)
			if t < 0 {
				t = 0
			}
			if t > 1 {
				t = 1
			}
			freqMask = 1.0 - t
		}

		// Combined frequency
		freqData[i] = float32((freqInter * freqExtra) / (freqInter*freqMask + freqExtra*(1-freqMask)))
	}

	return mlx.NewArray(freqData, []int32{halfDims}), mscale
}

// initYarn initializes YaRN RoPE if configured
func (a *Attention) initYarn(cfg *Config) {
	a.YarnMscale = 1.0
	if cfg.RopeScaling != nil && cfg.RopeScaling.RopeType == "yarn" {
		a.YarnFreqs, a.YarnMscale = ComputeYarnFreqs(
			cfg.HeadDim,
			cfg.RopeTheta,
			cfg.RopeScaling.Factor,
			cfg.RopeScaling.OriginalMaxPositionEmbeddings,
			cfg.RopeScaling.BetaFast,
			cfg.RopeScaling.BetaSlow,
		)
	}
}

// Forward performs the attention forward pass
func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, mask *mlx.Array, maskMode string, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape via AsStrided: [B, L, n_heads * head_dim] -> [B, n_heads, L, head_dim]
	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	if a.YarnFreqs != nil {
		if a.YarnMscale != 1.0 {
			q = mlx.MulScalar(q, a.YarnMscale)
		}
		q = mlx.RoPEWithFreqs(q, a.YarnFreqs, int(cfg.HeadDim), false, 1.0, offset)
		k = mlx.RoPEWithFreqs(k, a.YarnFreqs, int(cfg.HeadDim), false, 1.0, offset)
	} else {
		q = mlx.RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)
		k = mlx.RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)
	}

	if c != nil {
		k, v = c.Update(k, v, int(L))
	}

	out := mlx.ScaledDotProductAttentionWithSinks(q, k, v, cfg.Scale, maskMode, mask, a.Sinks)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// CreateSlidingWindowMask creates a causal mask with sliding window
// Mirrors mlx-lm's create_causal_mask with window_size
func CreateSlidingWindowMask(seqLen, queryStart, keyStart, keyLen, windowSize int) *mlx.Array {
	// Build mask aligned to actual cache length (may be rotated)
	rinds := mlx.Arange(float32(keyStart), float32(keyStart+keyLen), 1)
	linds := mlx.Arange(float32(queryStart), float32(queryStart+seqLen), 1)

	linds = mlx.ExpandDims(linds, 1)
	rinds = mlx.ExpandDims(rinds, 0)

	causalMask := mlx.GreaterEqual(linds, rinds)
	windowLimit := mlx.AddScalar(rinds, float32(windowSize))
	windowMask := mlx.LessArray(linds, windowLimit)

	return mlx.LogicalAnd(causalMask, windowMask)
}
