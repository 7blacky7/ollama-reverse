//go:build mlx

// mlx_quant.go - Quantisierung und MoE Operationen
//
// Enthaelt:
// - Quantize, Dequantize, QuantizedMatmul
// - GatherMM, GatherQMM (fuer MoE)

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

// Quantize quantizes weights to specified bits per element.
// Returns (quantized_weights, scales, biases).
// groupSize: number of elements quantized together (default 64)
// bits: bits per element, 2, 4, or 8 (default 4)
// mode: "affine" (default), "mxfp4", or "mxfp8"
// Note: mxfp8 mode returns nil biases (only weights and scales)
func Quantize(w *Array, groupSize, bits int, mode string) (weights, scales, biases *Array) {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	res := C.mlx_vector_array_new()
	C.mlx_quantize(&res, w.c, optGroupSize, optBits, cMode, C.default_stream())

	// Result is a vector of arrays: [weights, scales, biases?]
	// mxfp8 mode returns only 2 elements (no biases)
	vecSize := int(C.mlx_vector_array_size(res))
	var w0, w1, w2 C.mlx_array
	C.mlx_vector_array_get(&w0, res, 0)
	C.mlx_vector_array_get(&w1, res, 1)
	if vecSize >= 3 {
		C.mlx_vector_array_get(&w2, res, 2)
	}
	C.mlx_vector_array_free(res)

	if vecSize >= 3 {
		return newArray(w0), newArray(w1), newArray(w2)
	}
	return newArray(w0), newArray(w1), nil
}

// Dequantize reconstructs weights from quantized form.
// groupSize: number of elements quantized together (default 64)
// bits: bits per element, 2, 4, or 8 (default 4)
// mode: "affine" (default) or "mxfp4"
func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	optDtype := C.mlx_optional_dtype{has_value: false}

	var b C.mlx_array
	if biases != nil {
		b = biases.c
	}

	res := C.mlx_array_new()
	C.mlx_dequantize(&res, w.c, scales.c, b, optGroupSize, optBits, cMode, optDtype, C.default_stream())
	return newArray(res)
}

// QuantizedMatmul performs matrix multiplication with quantized weights.
// x: input tensor [batch..., in_features]
// w: quantized weights
// scales, biases: from Quantize
// transpose: if true, compute x @ w.T (typical for Linear layers)
// groupSize, bits, mode: must match what was used in Quantize
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}

	var b C.mlx_array
	if biases != nil {
		b = biases.c
	}

	res := C.mlx_array_new()
	C.mlx_quantized_matmul(&res, x.c, w.c, scales.c, b, C._Bool(transpose), optGroupSize, optBits, cMode, C.default_stream())
	return newArray(res)
}

// GatherMM performs gather matrix multiplication for MoE
// a: input, b: weight matrices
// lhsIndices, rhsIndices: optional expert selection indices (nil for none)
func GatherMM(a, b *Array, lhsIndices, rhsIndices *Array, sortedIndices bool) *Array {
	var lhs, rhs C.mlx_array
	if lhsIndices != nil {
		lhs = lhsIndices.c
	}
	if rhsIndices != nil {
		rhs = rhsIndices.c
	}
	res := C.mlx_array_new()
	C.mlx_gather_mm(&res, a.c, b.c, lhs, rhs, C._Bool(sortedIndices), C.default_stream())
	return newArray(res)
}

// GatherQMM performs quantized gather matrix multiplication for MoE
// Used for MXFP4 and other quantized MoE inference
func GatherQMM(x, w, scales *Array, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, mode string, sortedIndices bool) *Array {
	var b, lhs, rhs C.mlx_array
	if biases != nil {
		b = biases.c
	}
	if lhsIndices != nil {
		lhs = lhsIndices.c
	}
	if rhsIndices != nil {
		rhs = rhsIndices.c
	}
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	res := C.mlx_array_new()
	C.mlx_gather_qmm(&res, x.c, w.c, scales.c, b, lhs, rhs, C._Bool(transpose), optGroupSize, optBits, cMode, C._Bool(sortedIndices), C.default_stream())
	return newArray(res)
}
