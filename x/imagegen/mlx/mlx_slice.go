//go:build mlx

// mlx_slice.go - Slicing und Update Operationen
//
// Enthaelt:
// - Slice, SliceStride, SliceAxis
// - SliceUpdate, SliceUpdateInplace

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

// Slice slices the array
func Slice(a *Array, start, stop []int32) *Array {
	n := len(start)
	cStart := make([]C.int, n)
	cStop := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = 1
	}
	res := C.mlx_array_new()
	C.mlx_slice(&res, a.c, &cStart[0], C.size_t(n), &cStop[0], C.size_t(n), &cStrides[0], C.size_t(n), C.default_stream())
	return newArray(res)
}

// SliceStride slices with start:stop:stride like Python a[start:stop:stride]
func SliceStride(a *Array, start, stop, strides []int32) *Array {
	cStart := make([]C.int, len(start))
	cStop := make([]C.int, len(stop))
	cStrides := make([]C.int, len(strides))
	for i := range start {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = C.int(strides[i])
	}
	res := C.mlx_array_new()
	C.mlx_slice(&res, a.c, &cStart[0], C.size_t(len(start)), &cStop[0], C.size_t(len(stop)), &cStrides[0], C.size_t(len(strides)), C.default_stream())
	return newArray(res)
}

// SliceAxis extracts a slice along a specific axis
func SliceAxis(a *Array, axis int, start, stop int32) *Array {
	shape := a.Shape()
	starts := make([]int32, len(shape))
	stops := make([]int32, len(shape))
	for i := range shape {
		if i == axis {
			starts[i] = start
			stops[i] = stop
		} else {
			starts[i] = 0
			stops[i] = shape[i]
		}
	}
	return Slice(a, starts, stops)
}

// SliceUpdate updates a slice of the array with new values
func SliceUpdate(a, update *Array, start, stop []int32) *Array {
	n := len(start)
	cStart := make([]C.int, n)
	cStop := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = 1
	}
	res := C.mlx_array_new()
	C.mlx_slice_update(&res, a.c, update.c, &cStart[0], C.size_t(n), &cStop[0], C.size_t(n), &cStrides[0], C.size_t(n), C.default_stream())
	return newArray(res)
}

// SliceUpdateInplace updates a slice and returns a new array.
// Note: NOT in-place - MLX arrays are immutable.
func SliceUpdateInplace(a, update *Array, start, stop []int32) *Array {
	return SliceUpdate(a, update, start, stop)
}
