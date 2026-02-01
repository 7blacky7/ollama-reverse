//go:build mlx

// Package mlx - Erweiterte Array-Operationen (NN-Layers, Transformationen)

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"log/slog"
	"unsafe"

	"github.com/ollama/ollama/x/ml"
	"github.com/x448/float16"
)

// Matmul führt Matrix-Multiplikation durch
func (a *Array) Matmul(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_matmul(&r, a.a, a2.(*Array).a, ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// RMSNorm berechnet RMS-Normalisierung
func (a *Array) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_rms_norm(&r, a.a, w.(*Array).a, C.float(eps), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// LayerNorm berechnet Layer-Normalisierung
func (a *Array) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_layer_norm(&r, a.a, w.(*Array).a, b.(*Array).a, C.float(eps), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// L2Norm berechnet L2-Normalisierung (nicht implementiert)
func (a *Array) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

// AvgPool2D führt Average-Pooling durch (nicht implementiert)
func (t Array) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

// RoPE implementiert Rotary Positional Encoding
func (a *Array) RoPE(ctx ml.Context, dims int, traditional bool, scale float32, offset int, options ...func(*ml.RoPEOptions)) ml.Tensor {
	opts := ml.RoPEOptions{}
	for _, option := range options {
		option(&opts)
	}
	var r C.mlx_array
	var base C.mlx_optional_float
	var freqs C.mlx_array
	if opts.Base != nil {
		base.value = C.float(*opts.Base)
		base.has_value = true
	}
	if opts.Freqs != nil {
		freqs = opts.Freqs.(*Array).a
	}
	C.mlx_fast_rope(&r, a.a, C.int(dims), C._Bool(traditional), base,
		C.float(scale), C.int(offset), freqs, ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// ScaledDotProductAttention implementiert Multi-Head Attention
// queries: [B, N_q, T_q, D], keys/values: [B, N_kv, T_kv, D]
func (queries *Array) ScaledDotProductAttention(ctx ml.Context, keys, values ml.Tensor, scale float64, maskMode string, mask ml.Tensor, sinks ml.Tensor) ml.Tensor {
	var r, s C.mlx_array
	if sinks != nil {
		s = sinks.(*Array).a
	}
	maskModeC := C.CString(maskMode)
	defer C.free(unsafe.Pointer(maskModeC))
	var maskArr C.mlx_array
	if mask != nil {
		maskArr = mask.(*Array).a
	}
	C.mlx_fast_scaled_dot_product_attention(&r, queries.a, keys.(*Array).a, values.(*Array).a,
		C.float(scale), maskModeC, maskArr, s, ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// TakeAxes nimmt Elemente entlang einer Achse
func (a *Array) TakeAxes(ctx ml.Context, indicies ml.Tensor, axes int) ml.Tensor {
	var r C.mlx_array
	C.mlx_take_axis(&r, a.a, indicies.(*Array).a, C.int(axes), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// GELU berechnet die GELU-Aktivierungsfunktion: x * sigmoid(1.702 * x)
func (a *Array) GELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	const geluCoefficient = 1.702
	u16s := []float16.Float16{float16.Fromfloat32(geluCoefficient)}
	cshape := []C.int{1}
	f := C.mlx_array_new_data(unsafe.Pointer(&u16s[0]), &cshape[0], 1, C.MLX_FLOAT16)
	defer C.mlx_array_free(f)
	var r1, r2, r3 C.mlx_array
	C.mlx_multiply(&r1, a.a, f, ctx.(*Context).stream)
	defer C.mlx_array_free(r1)
	C.mlx_sigmoid(&r2, r1, ctx.(*Context).stream)
	defer C.mlx_array_free(r2)
	C.mlx_multiply(&r3, a.a, r2, ctx.(*Context).stream)
	if len(up) > 0 {
		var r4 C.mlx_array
		defer C.mlx_array_free(r3)
		C.mlx_multiply(&r4, r3, up[0].(*Array).a, ctx.(*Context).stream)
		return newArray(ctx.(*Context), r4)
	}
	return newArray(ctx.(*Context), r3)
}

// AsStrided erstellt eine View mit gegebenen Shape/Strides/Offset
func (a *Array) AsStrided(ctx ml.Context, shape, strides []int, offset int) ml.Tensor {
	var r C.mlx_array
	sh := make([]C.int, len(shape))
	st := make([]C.int64_t, len(strides))
	var sh0 *C.int
	var st0 *C.int64_t
	for i, s := range shape {
		sh[i] = C.int(s)
	}
	for i, s := range strides {
		st[i] = C.int64_t(s)
	}
	if len(sh) > 0 {
		sh0 = (*C.int)(unsafe.Pointer(&sh[0]))
	}
	if len(st) > 0 {
		st0 = (*C.int64_t)(unsafe.Pointer(&st[0]))
	}
	C.mlx_as_strided(&r, a.a, sh0, C.size_t(len(sh)), st0, C.size_t(len(st)),
		C.size_t(offset), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// Reshape ändert die Form des Arrays
func (a *Array) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	var r C.mlx_array
	C.mlx_reshape(&r, a.a, &cshape[0], C.size_t(len(cshape)), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// Transpose transponiert das Array
func (a *Array) Transpose(ctx ml.Context, shape ...int) ml.Tensor {
	ndim := min(C.mlx_array_ndim(a.a), C.size_t(len(shape)))
	var r C.mlx_array
	sh := make([]C.int, ndim)
	for i := range ndim {
		sh[i] = (C.int)(shape[i])
		if int(sh[i]) >= int(ndim) {
			slog.Error("Permute error", "tensor", a, "shape", shape)
			panic("invalid pemute call")
		}
	}
	if len(sh) > 0 {
		C.mlx_transpose_axes(&r, a.a, &sh[0], ndim, ctx.(*Context).stream)
	} else {
		C.mlx_transpose(&r, a.a, ctx.(*Context).stream)
	}
	return newArray(ctx.(*Context), r)
}

// Contiguous macht das Array zusammenhängend im Speicher
func (a *Array) Contiguous(ctx ml.Context, allowColMajor bool) ml.Tensor {
	var r C.mlx_array
	C.mlx_contiguous(&r, a.a, (C._Bool)(allowColMajor), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// Conv2D führt 2D-Faltung durch
func (input *Array) Conv2D(ctx ml.Context, weight ml.Tensor, stride0, stride1, padding0, padding1, dilation0, dilation1, groups int) ml.Tensor {
	var r C.mlx_array
	C.mlx_conv2d(&r, input.a, weight.(*Array).a,
		C.int(stride0), C.int(stride1), C.int(padding0), C.int(padding1),
		C.int(dilation0), C.int(dilation1), C.int(groups), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// Conv3D führt 3D-Faltung durch
func (input *Array) Conv3D(ctx ml.Context, weight ml.Tensor, stride0, stride1, stride2, padding0, padding1, padding2, dilation0, dilation1, dilation2, groups int) ml.Tensor {
	var r C.mlx_array
	C.mlx_conv3d(&r, input.a, weight.(*Array).a,
		C.int(stride0), C.int(stride1), C.int(stride2),
		C.int(padding0), C.int(padding1), C.int(padding2),
		C.int(dilation0), C.int(dilation1), C.int(dilation2),
		C.int(groups), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}
