//go:build mlx

// mlx_compare.go - Vergleichs- und Toleranz-Operationen
//
// Enthaelt:
// - AllClose, AllCloseEqualNaN
// - ArrayEqual, ArrayEqualNaN
// - IsClose, IsCloseEqualNaN
// - GreaterEqual, LessArray, LessScalar, LogicalAnd

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

// GreaterEqual returns element-wise a >= b
func GreaterEqual(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_greater_equal(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// LessArray returns element-wise a < b
func LessArray(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_less(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// LessScalar returns element-wise a < scalar
func LessScalar(a *Array, s float32) *Array {
	scalar := C.mlx_array_new_float(C.float(s))
	res := C.mlx_array_new()
	C.mlx_less(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// LogicalAnd returns element-wise a && b
func LogicalAnd(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_logical_and(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AllClose returns true if all elements of a and b are within tolerance.
// |a - b| <= atol + rtol * |b|
func AllClose(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_allclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(false), C.default_stream())
	return newArray(res)
}

// AllCloseEqualNaN is like AllClose but treats NaN as equal to NaN.
func AllCloseEqualNaN(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_allclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(true), C.default_stream())
	return newArray(res)
}

// ArrayEqual returns true if arrays have same shape and all elements are equal.
func ArrayEqual(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_array_equal(&res, a.c, b.c, C.bool(false), C.default_stream())
	return newArray(res)
}

// ArrayEqualNaN is like ArrayEqual but treats NaN as equal to NaN.
func ArrayEqualNaN(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_array_equal(&res, a.c, b.c, C.bool(true), C.default_stream())
	return newArray(res)
}

// IsClose returns element-wise bool array indicating if values are within tolerance.
// |a - b| <= atol + rtol * |b|
func IsClose(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_isclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(false), C.default_stream())
	return newArray(res)
}

// IsCloseEqualNaN is like IsClose but treats NaN as equal to NaN.
func IsCloseEqualNaN(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_isclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(true), C.default_stream())
	return newArray(res)
}

// Where selects elements: condition ? a : b
func Where(condition, a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_where(&res, condition.c, a.c, b.c, C.default_stream())
	return newArray(res)
}
