//go:build mlx

// mlx_math.go - Mathematische Operationen
//
// Enthaelt:
// - Elementweise Operationen (Add, Sub, Mul, Div, Sqrt, Exp, etc.)
// - Scalar Operationen (AddScalar, MulScalar, DivScalar)
// - AsType, ToBFloat16

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

// Add adds two arrays element-wise
func Add(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_add(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AddRaw is like Add - kept for API compatibility
func AddRaw(a, b *Array) *Array { return Add(a, b) }

// Sub subtracts two arrays element-wise
func Sub(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_subtract(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Mul multiplies two arrays element-wise
func Mul(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Div divides two arrays element-wise
func Div(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Matmul performs matrix multiplication
func Matmul(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_matmul(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AddMM computes: result = beta*c + alpha*(a @ b)
func AddMM(c, a, b *Array, alpha, beta float32) *Array {
	res := C.mlx_array_new()
	C.mlx_addmm(&res, c.c, a.c, b.c, C.float(alpha), C.float(beta), C.default_stream())
	return newArray(res)
}

// Linear performs matrix multiplication: a @ weight
func Linear(a, weight *Array) *Array { return Matmul(a, weight) }

// Sqrt computes element-wise square root
func Sqrt(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sqrt(&res, a.c, C.default_stream())
	return newArray(res)
}

// RSqrt computes element-wise reciprocal square root
func RSqrt(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_rsqrt(&res, a.c, C.default_stream())
	return newArray(res)
}

// Erf computes element-wise error function
func Erf(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_erf(&res, a.c, C.default_stream())
	return newArray(res)
}

// Exp computes element-wise exponential
func Exp(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_exp(&res, a.c, C.default_stream())
	return newArray(res)
}

// Log computes element-wise natural logarithm
func Log(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_log(&res, a.c, C.default_stream())
	return newArray(res)
}

// Sin computes element-wise sine
func Sin(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sin(&res, a.c, C.default_stream())
	return newArray(res)
}

// Cos computes element-wise cosine
func Cos(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_cos(&res, a.c, C.default_stream())
	return newArray(res)
}

// Neg negates the array
func Neg(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_negative(&res, a.c, C.default_stream())
	return newArray(res)
}

// Abs computes element-wise absolute value
func Abs(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_abs(&res, a.c, C.default_stream())
	return newArray(res)
}

// Square computes element-wise square
func Square(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_square(&res, a.c, C.default_stream())
	return newArray(res)
}

// Pow raises a to the power of b element-wise
func Pow(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_power(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Max computes element-wise maximum
func Max(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_maximum(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Min computes element-wise minimum
func Min(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_minimum(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// scalarWithDtype creates a scalar array matching the dtype of a
func scalarWithDtype(s float32, a *Array) C.mlx_array {
	f32 := C.mlx_array_new_float(C.float(s))
	dtype := a.Dtype()
	if dtype == DtypeFloat32 {
		return f32
	}
	casted := C.mlx_array_new()
	C.mlx_astype(&casted, f32, C.mlx_dtype(dtype), C.default_stream())
	C.mlx_array_free(f32)
	return casted
}

// AddScalar adds a scalar to an array
func AddScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_add(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// MulScalar multiplies an array by a scalar
func MulScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// DivScalar divides an array by a scalar
func DivScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// DivScalarInt divides an int array by an int scalar
func DivScalarInt(a *Array, s int32) *Array {
	scalar := C.mlx_array_new_int(C.int(s))
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// FloorDivideScalar performs integer floor division (a // s)
func FloorDivideScalar(a *Array, s int32) *Array {
	scalar := C.mlx_array_new_int(C.int(s))
	res := C.mlx_array_new()
	C.mlx_floor_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// AsType casts an array to a different dtype
func AsType(a *Array, dtype Dtype) *Array {
	res := C.mlx_array_new()
	C.mlx_astype(&res, a.c, C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// ToBFloat16 casts an array to bfloat16
func ToBFloat16(a *Array) *Array { return AsType(a, DtypeBFloat16) }
