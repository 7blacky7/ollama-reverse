//go:build mlx

// mlx_nn.go - Neural Network Operationen
//
// Enthaelt:
// - Aktivierungsfunktionen (Softmax, Sigmoid, ReLU, SiLU, GELU, Tanh)
// - Normalisierung (RMSNorm, LayerNorm)
// - Attention (ScaledDotProductAttention, RoPE)
// - Embedding und Gather

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>

static mlx_stream _default_stream = {0};

static inline mlx_stream default_stream() {
    if (_default_stream.ctx == NULL) {
        _default_stream = mlx_default_gpu_stream_new();
    }
    return _default_stream;
}
*/
import "C"
import "unsafe"

// Softmax computes softmax along an axis
func Softmax(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_softmax_axis(&res, a.c, C.int(axis), false, C.default_stream())
	return newArray(res)
}

// Take gathers elements along an axis using indices
func Take(a *Array, indices *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_take_axis(&res, a.c, indices.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Argsort returns indices that would sort the array along an axis
func Argsort(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_argsort_axis(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Sigmoid computes element-wise sigmoid
func Sigmoid(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sigmoid(&res, a.c, C.default_stream())
	return newArray(res)
}

// ReLU computes element-wise ReLU: max(0, x)
func ReLU(a *Array) *Array {
	// ReLU = maximum(x, 0) - mlx-c doesn't have mlx_relu, but we can use maximum
	zero := C.mlx_array_new_float(0.0)
	res := C.mlx_array_new()
	C.mlx_maximum(&res, a.c, zero, C.default_stream())
	C.mlx_array_free(zero)
	return newArray(res)
}

// SiLU computes element-wise SiLU (Swish): x * sigmoid(x)
func SiLU(a *Array) *Array {
	// SiLU = x * sigmoid(x)
	sig := C.mlx_array_new()
	C.mlx_sigmoid(&sig, a.c, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, sig, C.default_stream())
	C.mlx_array_free(sig)
	return newArray(res)
}

// GELU computes element-wise GELU (Gaussian Error Linear Unit)
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
func GELU(a *Array) *Array {
	sqrt2 := C.mlx_array_new_float(1.4142135623730951)
	scaled := C.mlx_array_new()
	C.mlx_divide(&scaled, a.c, sqrt2, C.default_stream())
	erfd := C.mlx_array_new()
	C.mlx_erf(&erfd, scaled, C.default_stream())
	one := C.mlx_array_new_float(1.0)
	erfdPlusOne := C.mlx_array_new()
	C.mlx_add(&erfdPlusOne, erfd, one, C.default_stream())
	half := C.mlx_array_new_float(0.5)
	halfErfdPlusOne := C.mlx_array_new()
	C.mlx_multiply(&halfErfdPlusOne, half, erfdPlusOne, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, halfErfdPlusOne, C.default_stream())
	C.mlx_array_free(sqrt2)
	C.mlx_array_free(scaled)
	C.mlx_array_free(erfd)
	C.mlx_array_free(one)
	C.mlx_array_free(erfdPlusOne)
	C.mlx_array_free(half)
	C.mlx_array_free(halfErfdPlusOne)
	return newArray(res)
}

// Tanh computes element-wise tanh
func Tanh(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_tanh(&res, a.c, C.default_stream())
	return newArray(res)
}

// RMSNorm computes RMS normalization using mlx.fast
func RMSNorm(x, weight *Array, eps float32) *Array {
	res := C.mlx_array_new()
	C.mlx_fast_rms_norm(&res, x.c, weight.c, C.float(eps), C.default_stream())
	return newArray(res)
}

// RMSNormNoWeight applies RMS normalization without a weight
// x * rsqrt(mean(x^2) + eps)
// Uses mlx_fast_rms_norm with ones weight for f32 accumulation precision
func RMSNormNoWeight(x *Array, eps float32) *Array {
	// Create weight of ones matching last dimension
	lastDim := x.Shape()[len(x.Shape())-1]
	ones := AsType(Full(1.0, lastDim), x.Dtype())
	return RMSNorm(x, ones, eps)
}

// LayerNorm applies layer normalization without learnable params
// (x - mean) / sqrt(var + eps)
func LayerNorm(x *Array, eps float32) *Array {
	return LayerNormWithWeightBias(x, nil, nil, eps)
}

// LayerNormWithWeightBias computes layer normalization using mlx.fast
// weight and bias can be nil for elementwise_affine=False
func LayerNormWithWeightBias(x, weight, bias *Array, eps float32) *Array {
	res := C.mlx_array_new()
	var wc, bc C.mlx_array
	if weight != nil {
		wc = weight.c
	}
	if bias != nil {
		bc = bias.c
	}
	C.mlx_fast_layer_norm(&res, x.c, wc, bc, C.float(eps), C.default_stream())
	return newArray(res)
}

// RoPE applies rotary position embeddings using mlx.fast
func RoPE(x *Array, dims int, traditional bool, base, scale float32, offset int) *Array {
	res := C.mlx_array_new()
	optBase := C.mlx_optional_float{value: C.float(base), has_value: true}
	C.mlx_fast_rope(&res, x.c, C.int(dims), C._Bool(traditional), optBase, C.float(scale), C.int(offset), C.mlx_array{}, C.default_stream())
	return newArray(res)
}

// RoPEWithFreqs applies rotary position embeddings with custom frequencies (for YaRN)
// freqs is required - use RoPE() if you don't have custom frequencies
func RoPEWithFreqs(x, freqs *Array, dims int, traditional bool, scale float32, offset int) *Array {
	res := C.mlx_array_new()
	optBase := C.mlx_optional_float{has_value: false} // No base when using freqs
	C.mlx_fast_rope(&res, x.c, C.int(dims), C._Bool(traditional), optBase, C.float(scale), C.int(offset), freqs.c, C.default_stream())
	return newArray(res)
}

// EmbeddingLookup performs embedding lookup (gathers from table)
// table: [vocab_size, hidden_size], indices: [batch, seq_len]
// returns: [batch, seq_len, hidden_size]
func EmbeddingLookup(table, indices *Array) *Array {
	return Take(table, indices, 0)
}

// Gather gathers elements using indices - simplified to use take axis 0
func Gather(a, indices *Array) *Array {
	return Take(a, indices, 0)
}

// ScaledDotProductAttention computes optimized attention using GPU kernel
// Q, K, V should be [batch, heads, seq, head_dim]
func ScaledDotProductAttention(q, k, v *Array, scale float32, causalMask bool) *Array {
	res := C.mlx_array_new()
	maskMode := "" // empty string for no mask
	if causalMask {
		maskMode = "causal"
	}
	cMaskMode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cMaskMode))
	C.mlx_fast_scaled_dot_product_attention(&res, q.c, k.c, v.c, C.float(scale), cMaskMode, C.mlx_array{}, C.mlx_array{}, C.default_stream())
	return newArray(res)
}

// ScaledDotProductAttentionWithSinks computes attention with sinks support
// maskMode: "causal", "sliding_window", or "" for none
// mask: optional attention mask array (nil for none)
// sinks: attention sinks array (nil for none)
func ScaledDotProductAttentionWithSinks(q, k, v *Array, scale float32, maskMode string, mask, sinks *Array) *Array {
	res := C.mlx_array_new()
	cMaskMode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cMaskMode))
	var maskH, sinksH C.mlx_array
	if mask != nil {
		maskH = mask.c
	}
	if sinks != nil {
		sinksH = sinks.c
	}
	C.mlx_fast_scaled_dot_product_attention(&res, q.c, k.c, v.c, C.float(scale), cMaskMode, maskH, sinksH, C.default_stream())
	return newArray(res)
}

// TopK returns the k largest elements along an axis
func TopK(a *Array, k int, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_topk_axis(&res, a.c, C.int(k), C.int(axis), C.default_stream())
	return newArray(res)
}

// Argpartition returns indices for partial sort (k-th smallest first)
func Argpartition(a *Array, kth int, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_argpartition_axis(&res, a.c, C.int(kth), C.int(axis), C.default_stream())
	return newArray(res)
}

// TakeAlongAxis takes elements from array using indices along axis
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_take_along_axis(&res, a.c, indices.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// PutAlongAxis puts values into array at indices along axis
func PutAlongAxis(a, indices, values *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_put_along_axis(&res, a.c, indices.c, values.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// SampleArgmax gets the last logit position and returns argmax (fused operation)
func SampleArgmax(logits *Array) int32 {
	result := Argmax(logits, -1, false)
	return result.ItemInt32()
}

// ArgmaxKeepArray returns argmax as an Array (for pipelining, no sync)
// This is like mlx-lm's sampler that returns y as an array, not .item()
func ArgmaxKeepArray(logits *Array) *Array {
	// For greedy decoding: logits shape is [1, 1, vocab]
	// We want argmax over vocab dimension, return shape []
	return Argmax(logits, -1, false)
}
