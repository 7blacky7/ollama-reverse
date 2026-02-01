//go:build mlx

// mlx_create.go - Array Erstellung
//
// Enthaelt:
// - NewArray, NewArrayInt32, NewArrayFloat32
// - Zeros, ZerosLike, Ones, Full, FullDtype
// - Arange, Linspace, ArangeInt
// - NewArrayFromBytes, NewScalarArray

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

func int32ToCInt(s []int32) *C.int {
	if len(s) == 0 {
		return nil
	}
	return (*C.int)(unsafe.Pointer(&s[0]))
}

// NewArray creates a new MLX array from float32 data
func NewArray(data []float32, shape []int32) *Array {
	handle := C.mlx_array_new_data(
		unsafe.Pointer(&data[0]),
		int32ToCInt(shape),
		C.int(len(shape)),
		C.MLX_FLOAT32,
	)
	return newArray(handle)
}

// NewArrayInt32 creates a new MLX array from int32 data
func NewArrayInt32(data []int32, shape []int32) *Array {
	handle := C.mlx_array_new_data(
		unsafe.Pointer(&data[0]),
		int32ToCInt(shape),
		C.int(len(shape)),
		C.MLX_INT32,
	)
	return newArray(handle)
}

// NewArrayFloat32 creates a new float32 array from data
func NewArrayFloat32(data []float32, shape []int32) *Array {
	return NewArray(data, shape)
}

// Zeros creates an array of zeros with optional dtype (default float32)
func Zeros(shape []int32, dtype ...Dtype) *Array {
	res := C.mlx_array_new()
	dt := DtypeFloat32
	if len(dtype) > 0 {
		dt = dtype[0]
	}
	C.mlx_zeros(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dt), C.default_stream())
	return newArray(res)
}

// ZerosLike creates a zeros array with the same dtype as a.
// If shape is provided, uses that shape; otherwise uses a's shape.
func ZerosLike(a *Array, shape ...int32) *Array {
	res := C.mlx_array_new()
	if len(shape) == 0 {
		C.mlx_zeros_like(&res, a.c, C.default_stream())
	} else {
		dtype := a.Dtype()
		C.mlx_zeros(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dtype), C.default_stream())
	}
	return newArray(res)
}

// Ones creates an array of ones
func Ones(shape ...int32) *Array {
	res := C.mlx_array_new()
	C.mlx_ones(&res, int32ToCInt(shape), C.size_t(len(shape)), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// Full creates an array filled with a value
func Full(value float32, shape ...int32) *Array {
	vals := C.mlx_array_new_float(C.float(value))
	res := C.mlx_array_new()
	C.mlx_full(&res, int32ToCInt(shape), C.size_t(len(shape)), vals, C.MLX_FLOAT32, C.default_stream())
	C.mlx_array_free(vals)
	return newArray(res)
}

// FullDtype creates an array filled with a value with specific dtype
func FullDtype(value float32, dtype Dtype, shape ...int32) *Array {
	intShape := make([]C.int, len(shape))
	for i, s := range shape {
		intShape[i] = C.int(s)
	}
	vals := C.mlx_array_new_float(C.float(value))
	res := C.mlx_array_new()
	C.mlx_full(&res, &intShape[0], C.size_t(len(shape)), vals, C.mlx_dtype(dtype), C.default_stream())
	C.mlx_array_free(vals)
	return newArray(res)
}

// Arange creates a range of values
func Arange(start, stop, step float32) *Array {
	res := C.mlx_array_new()
	C.mlx_arange(&res, C.double(start), C.double(stop), C.double(step), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// ArangeInt creates an array with values from start to stop with step and specified dtype
func ArangeInt(start, stop, step int32, dtype Dtype) *Array {
	res := C.mlx_array_new()
	C.mlx_arange(&res, C.double(start), C.double(stop), C.double(step), C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// Linspace creates evenly spaced values
func Linspace(start, stop float32, steps int32) *Array {
	res := C.mlx_array_new()
	C.mlx_linspace(&res, C.double(start), C.double(stop), C.int(steps), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// NewArrayFromBytes creates an array from raw bytes (for safetensors)
func NewArrayFromBytes(data []byte, shape []int32, dtype Dtype) *Array {
	cData := unsafe.Pointer(&data[0])
	intShape := make([]C.int, len(shape))
	for i, s := range shape {
		intShape[i] = C.int(s)
	}
	handle := C.mlx_array_new_data(cData, &intShape[0], C.int(len(shape)), C.mlx_dtype(dtype))
	return newArray(handle)
}

// NewScalarArray creates a true 0-dimensional scalar array from a float32 value
func NewScalarArray(value float32) *Array {
	return newArray(C.mlx_array_new_float(C.float(value)))
}

// Tri creates a lower triangular matrix
func Tri(n, m int32, k int) *Array {
	res := C.mlx_array_new()
	C.mlx_tri(&res, C.int(n), C.int(m), C.int(k), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}
