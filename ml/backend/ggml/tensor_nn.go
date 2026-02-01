// tensor_nn.go - Neuronale Netzwerk Operationen
// Enthält: Aktivierungen (GELU, SILU, RELU), Convolution, Attention, RoPE

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"cmp"
	"unsafe"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)

// Softmax berechnet Softmax
func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_soft_max(ctx.(*Context).ctx, t.t),
	}
}

// Sin berechnet Sinus elementweise
func (t *Tensor) Sin(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sin(ctx.(*Context).ctx, t.t),
	}
}

// Cos berechnet Kosinus elementweise
func (t *Tensor) Cos(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cos(ctx.(*Context).ctx, t.t),
	}
}

// Tanh berechnet Tangens Hyperbolicus inplace
func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_tanh_inplace(ctx.(*Context).ctx, t.t),
	}
}

// Sigmoid berechnet Sigmoid inplace
func (t *Tensor) Sigmoid(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sigmoid_inplace(ctx.(*Context).ctx, t.t),
	}
}

// RoPE führt Rotary Position Embedding durch
func (t *Tensor) RoPE(ctx ml.Context, positions ml.Tensor, ropeDim int, ropeBase, ropeScale float32, options ...func(*rope.Options)) ml.Tensor {
	opts := rope.Options{Factors: &Tensor{}}

	for _, option := range options {
		option(&opts)
	}

	dequant := t.t
	if C.ggml_is_quantized(t.t._type) {
		dequant = C.ggml_cast(ctx.(*Context).ctx, t.t, C.GGML_TYPE_F32)
	}

	var tt *C.struct_ggml_tensor
	if len(opts.MRoPE.Sections) > 0 {
		mropeSections := make([]C.int32_t, 4)
		for i, section := range opts.MRoPE.Sections {
			mropeSections[i] = C.int32_t(section)
		}

		tt = C.ggml_rope_multi(
			ctx.(*Context).ctx,
			dequant,
			positions.(*Tensor).t,
			opts.Factors.(*Tensor).t,
			C.int(ropeDim),
			unsafe.SliceData(mropeSections),
			C.int(opts.Type),
			cmp.Or(C.int(opts.YaRN.OriginalContextLength), 128<<10),
			C.float(ropeBase),
			C.float(ropeScale),
			C.float(opts.YaRN.ExtrapolationFactor),
			cmp.Or(C.float(opts.YaRN.AttentionFactor), 1),
			cmp.Or(C.float(opts.YaRN.BetaFast), 32),
			cmp.Or(C.float(opts.YaRN.BetaSlow), 1),
		)
	} else {
		tt = C.ggml_rope_ext(
			ctx.(*Context).ctx,
			dequant,
			positions.(*Tensor).t,
			opts.Factors.(*Tensor).t,
			C.int(ropeDim),
			C.int(opts.Type),
			cmp.Or(C.int(opts.YaRN.OriginalContextLength), 128<<10),
			C.float(ropeBase),
			C.float(ropeScale),
			C.float(opts.YaRN.ExtrapolationFactor),
			cmp.Or(C.float(opts.YaRN.AttentionFactor), 1),
			cmp.Or(C.float(opts.YaRN.BetaFast), 32),
			cmp.Or(C.float(opts.YaRN.BetaSlow), 1),
		)
	}
	return &Tensor{b: t.b, t: tt}
}

// IM2Col führt Image to Column Transformation durch
func (t *Tensor) IM2Col(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_im2col(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1), true, C.GGML_TYPE_F32),
	}
}

// GELU führt GELU-Aktivierung durch
func (t *Tensor) GELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_geglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_gelu_inplace(ctx.(*Context).ctx, t.t),
	}
}

// QuickGELU führt schnelle GELU-Aktivierung durch
func (t *Tensor) QuickGELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	var tt *C.struct_ggml_tensor
	if len(t2) > 0 {
		tt = C.ggml_geglu_quick_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t)
	} else {
		tt = C.ggml_gelu_quick_inplace(ctx.(*Context).ctx, t.t)
	}
	return &Tensor{b: t.b, t: tt}
}

// SILU führt SILU-Aktivierung durch
func (t *Tensor) SILU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_swiglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_silu_inplace(ctx.(*Context).ctx, t.t),
	}
}

// RELU führt RELU-Aktivierung durch
func (t *Tensor) RELU(ctx ml.Context, t2 ...ml.Tensor) ml.Tensor {
	if len(t2) > 0 {
		return &Tensor{
			b: t.b,
			t: C.ggml_reglu_split(ctx.(*Context).ctx, t.t, t2[0].(*Tensor).t),
		}
	}
	return &Tensor{
		b: t.b,
		t: C.ggml_relu_inplace(ctx.(*Context).ctx, t.t),
	}
}

// SILUAlphaLimit führt SILU mit Alpha und Limit durch
func (t *Tensor) SILUAlphaLimit(ctx ml.Context, up ml.Tensor, alpha, limit float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_swiglu_oai(ctx.(*Context).ctx, t.t, up.(*Tensor).t, C.float(alpha), C.float(limit)),
	}
}

// Conv2D führt 2D-Convolution durch
func (t *Tensor) Conv2D(ctx ml.Context, t2 ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_conv_2d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(s0), C.int(s1), C.int(p0), C.int(p1), C.int(d0), C.int(d1)),
	}
}

// Conv3D führt 3D-Convolution durch
func (t *Tensor) Conv3D(ctx ml.Context, t2 ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	var tt ml.Tensor = &Tensor{
		b: t.b,
		t: C.ggml_conv_3d(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int64_t(c), C.int(s0), C.int(s1), C.int(s2), C.int(p0), C.int(p1), C.int(p2), C.int(d0), C.int(d1), C.int(d2)),
	}

	tt = tt.Reshape(ctx, t.Dim(3)/c, t2.Dim(3)/c)
	return tt
}

// SSMConv führt SSM-Convolution durch
func (t *Tensor) SSMConv(ctx ml.Context, kernel ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_ssm_conv(ctx.(*Context).ctx, t.t, kernel.(*Tensor).t),
	}
}

// AvgPool2D führt 2D-Average-Pooling durch
func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_pool_2d(ctx.(*Context).ctx, t.t, C.GGML_OP_POOL_AVG, C.int(k), C.int(k), C.int(s), C.int(s), C.float(p), C.float(p)),
	}
}

// ScaledDotProductAttention führt Scaled Dot-Product Attention durch
func (t *Tensor) ScaledDotProductAttention(ctx ml.Context, key, value, mask, sinks ml.Tensor, vmla ml.Tensor, scale float64, cacheConfigApplied bool) ml.Tensor {
	// Transformationen falls Cache nicht geholfen hat
	if !cacheConfigApplied {
		cacheConfig := t.b.CacheConfig()

		if cacheConfig.PermutedV {
			value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
		}

		if mask != nil {
			if mask.DType() != cacheConfig.MaskDType {
				mask = mask.Cast(ctx, cacheConfig.MaskDType)
			}
		}
	}

	var kqMask *C.struct_ggml_tensor
	if mask != nil {
		kqMask = mask.(*Tensor).t
	}

	query := t.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)

	if t.b.flashAttention == ml.FlashAttentionEnabled {
		value = value.Permute(ctx, 0, 2, 1, 3)

		kqv := C.ggml_flash_attn_ext(ctx.(*Context).ctx, query.(*Tensor).t, key.(*Tensor).t, value.(*Tensor).t, kqMask, C.float(scale), 0, 0)
		if sinks != nil {
			C.ggml_flash_attn_ext_add_sinks(kqv, sinks.(*Tensor).t)
		}
		C.ggml_flash_attn_ext_set_prec(kqv, C.GGML_PREC_F32)

		if vmla != nil {
			var cur ml.Tensor = &Tensor{b: t.b, t: kqv}
			cur = cur.Permute(ctx, 0, 2, 1, 3)
			cur = vmla.Mulmat(ctx, cur)
			cur = cur.Permute(ctx, 0, 2, 1, 3)
			cur = cur.Contiguous(ctx)
			kqv = cur.(*Tensor).t
		}

		return &Tensor{b: t.b, t: kqv}
	}

	// Non-Flash-Attention Pfad
	kq := key.MulmatFullPrec(ctx, query)
	kq = &Tensor{
		b: t.b,
		t: C.ggml_soft_max_ext(ctx.(*Context).ctx, kq.(*Tensor).t, kqMask, C.float(scale), 0),
	}
	if sinks != nil {
		C.ggml_soft_max_add_sinks(kq.(*Tensor).t, sinks.(*Tensor).t)
	}

	kqv := value.Mulmat(ctx, kq)
	if vmla != nil {
		kqv = vmla.Mulmat(ctx, kqv)
	}

	return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
}
