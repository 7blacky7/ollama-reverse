//go:build mlx

// mlx_reduce.go - Reduktionen und Shape-Operationen
//
// Enthaelt:
// - Reduktionen (Sum, Mean, Var, Argmax, ReduceMax)
// - Shape-Operationen (Reshape, Transpose, ExpandDims, Squeeze, Flatten)
// - Concatenation, Tile, BroadcastTo, AsStrided, Clip

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

// Sum reduces along an axis
func Sum(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_sum_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// SumAll reduces the entire array to a scalar
func SumAll(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sum(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Mean reduces along an axis
func Mean(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_mean_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// MeanAll reduces the entire array to a scalar
func MeanAll(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_mean(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Var computes variance along an axis
func Var(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_var_axis(&res, a.c, C.int(axis), C._Bool(keepdims), 0, C.default_stream())
	return newArray(res)
}

// Argmax returns indices of maximum values along an axis
func Argmax(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_argmax_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// ArgmaxAll returns the index of the maximum element (flattened).
func ArgmaxAll(a *Array) int32 {
	cleanup()
	flat := C.mlx_array_new()
	C.mlx_flatten(&flat, a.c, 0, -1, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_argmax(&res, flat, false, C.default_stream())
	C.mlx_array_eval(res)
	var val C.int32_t
	C.mlx_array_item_int32(&val, res)
	C.mlx_array_free(flat)
	C.mlx_array_free(res)
	return int32(val)
}

// ReduceMax reduces array to max value over all dimensions.
func ReduceMax(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_max(&res, a.c, C.bool(false), C.default_stream())
	return newArray(res)
}

// Reshape reshapes the array
func Reshape(a *Array, shape ...int32) *Array {
	res := C.mlx_array_new()
	C.mlx_reshape(&res, a.c, int32ToCInt(shape), C.size_t(len(shape)), C.default_stream())
	return newArray(res)
}

// Transpose permutes the dimensions
func Transpose(a *Array, axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	res := C.mlx_array_new()
	C.mlx_transpose_axes(&res, a.c, &cAxes[0], C.size_t(len(axes)), C.default_stream())
	return newArray(res)
}

// AsStrided creates a view with custom strides.
func AsStrided(a *Array, shape []int32, strides []int64, offset int64) *Array {
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	cStrides := make([]C.int64_t, len(strides))
	for i, s := range strides {
		cStrides[i] = C.int64_t(s)
	}
	res := C.mlx_array_new()
	C.mlx_as_strided(&res, a.c, &cShape[0], C.size_t(len(shape)), &cStrides[0], C.size_t(len(strides)), C.size_t(offset), C.default_stream())
	return newArray(res)
}

// ExpandDims adds a dimension at the specified axis
func ExpandDims(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_expand_dims(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Squeeze removes a dimension at the specified axis
func Squeeze(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_squeeze_axis(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Flatten flattens the array to 1D
func Flatten(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_flatten(&res, a.c, 0, -1, C.default_stream())
	return newArray(res)
}

// FlattenRange flattens consecutive axes
func FlattenRange(a *Array, startAxis, endAxis int) *Array {
	res := C.mlx_array_new()
	C.mlx_flatten(&res, a.c, C.int(startAxis), C.int(endAxis), C.default_stream())
	return newArray(res)
}

// View reinterprets the array with a new dtype (no data copy)
func View(a *Array, dtype int) *Array {
	res := C.mlx_array_new()
	C.mlx_view(&res, a.c, C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// Contiguous returns a contiguous copy of the array
func Contiguous(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_contiguous(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Clip clips values to [min, max]. Pass nil for no bound.
func Clip(a *Array, aMin, aMax *Array) *Array {
	res := C.mlx_array_new()
	var minH, maxH C.mlx_array
	if aMin != nil {
		minH = aMin.c
	}
	if aMax != nil {
		maxH = aMax.c
	}
	C.mlx_clip(&res, a.c, minH, maxH, C.default_stream())
	return newArray(res)
}

// ClipScalar clips array values using scalar bounds
func ClipScalar(a *Array, minVal, maxVal float32, hasMin, hasMax bool) *Array {
	var minArr, maxArr C.mlx_array
	if hasMin {
		minArr = scalarWithDtype(minVal, a)
	}
	if hasMax {
		maxArr = scalarWithDtype(maxVal, a)
	}
	res := C.mlx_array_new()
	C.mlx_clip(&res, a.c, minArr, maxArr, C.default_stream())
	if hasMin {
		C.mlx_array_free(minArr)
	}
	if hasMax {
		C.mlx_array_free(maxArr)
	}
	return newArray(res)
}

// Concatenate concatenates arrays along an axis
func Concatenate(arrs []*Array, axis int) *Array {
	handles := make([]C.mlx_array, len(arrs))
	for i, arr := range arrs {
		handles[i] = arr.c
	}
	vec := C.mlx_vector_array_new_data(&handles[0], C.size_t(len(handles)))
	res := C.mlx_array_new()
	C.mlx_concatenate_axis(&res, vec, C.int(axis), C.default_stream())
	C.mlx_vector_array_free(vec)
	return newArray(res)
}

// Concat concatenates two arrays
func Concat(a, b *Array, axis int) *Array { return Concatenate([]*Array{a, b}, axis) }

// Tile repeats the array along each dimension
func Tile(a *Array, reps []int32) *Array {
	res := C.mlx_array_new()
	C.mlx_tile(&res, a.c, int32ToCInt(reps), C.size_t(len(reps)), C.default_stream())
	return newArray(res)
}

// BroadcastTo broadcasts an array to a given shape
func BroadcastTo(a *Array, shape []int32) *Array {
	res := C.mlx_array_new()
	C.mlx_broadcast_to(&res, a.c, int32ToCInt(shape), C.size_t(len(shape)), C.default_stream())
	return newArray(res)
}

// Cumsum computes cumulative sum along an axis
func Cumsum(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_cumsum(&res, a.c, C.int(axis), false, false, C.default_stream())
	return newArray(res)
}
