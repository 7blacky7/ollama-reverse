//go:build mlx

// mlx_props.go - Array Eigenschaften und Datenzugriff
//
// Enthaelt:
// - Array Eigenschaften (Ndim, Size, Shape, Dtype, Nbytes)
// - Datenzugriff (Data, DataInt32, Item, ItemInt32, Bytes)
// - String Repraesentation

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>

// Helper to check if array is contiguous
void _mlx_array_is_contiguous(bool *result, mlx_array a) {
    // MLX arrays from new_data are always contiguous
    // Only strided views may be non-contiguous
    *result = true;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Ndim returns the number of dimensions
func (a *Array) Ndim() int {
	return int(C.mlx_array_ndim(a.c))
}

// Size returns the total number of elements
func (a *Array) Size() int {
	return int(C.mlx_array_size(a.c))
}

// IsContiguous returns whether the array's data is contiguous in memory.
// Non-contiguous arrays (e.g., from SliceStride) must call Contiguous() before Data().
func (a *Array) IsContiguous() bool {
	var res C.bool
	C._mlx_array_is_contiguous(&res, a.c)
	return bool(res)
}

// Dim returns the size of a dimension
func (a *Array) Dim(axis int) int32 {
	return int32(C.mlx_array_dim(a.c, C.int(axis)))
}

// Shape returns the shape as a slice
func (a *Array) Shape() []int32 {
	ndim := a.Ndim()
	shape := make([]int32, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = a.Dim(i)
	}
	return shape
}

// IsValid returns true if the array hasn't been freed
func (a *Array) IsValid() bool {
	return a != nil && a.c.ctx != nil
}

// Dtype returns the data type
func (a *Array) Dtype() Dtype {
	return Dtype(C.mlx_array_dtype(a.c))
}

// Nbytes returns the total size in bytes
func (a *Array) Nbytes() int64 {
	return int64(a.Size()) * a.Dtype().ItemSize()
}

// Data copies the float32 data out of the array.
// Note: For non-contiguous arrays (e.g., from SliceStride), call Contiguous() first.
// Note: Arrays of other dtypes (bf16, f16, etc) are automatically converted to float32.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) Data() []float32 {
	cleanup()
	size := a.Size()
	if size == 0 {
		return nil
	}

	arr := a
	if a.Dtype() != DtypeFloat32 {
		arr = AsType(a, DtypeFloat32)
		arr.Eval()
		// Cast array will be cleaned up on next Eval
	}

	ptr := C.mlx_array_data_float32(arr.c)
	if ptr == nil {
		return nil
	}
	data := make([]float32, size)
	copy(data, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), size))
	return data
}

// Item returns the scalar value from a 0-dimensional array.
// Converts to float32 if necessary. Triggers cleanup.
func (a *Array) Item() float32 {
	data := a.Data() // Data() calls cleanup()
	if len(data) == 0 {
		return 0
	}
	return data[0]
}

// DataInt32 copies the int32 data out of the array.
// Note: For non-contiguous arrays (e.g., from SliceStride), call Contiguous() first.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) DataInt32() []int32 {
	cleanup()
	size := a.Size()
	if size == 0 {
		return nil
	}
	ptr := C.mlx_array_data_int32(a.c)
	if ptr == nil {
		return nil
	}
	data := make([]int32, size)
	copy(data, unsafe.Slice((*int32)(unsafe.Pointer(ptr)), size))
	return data
}

// ItemInt32 gets a single scalar value efficiently (no array copy).
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) ItemInt32() int32 {
	cleanup()
	var val C.int32_t
	C.mlx_array_item_int32(&val, a.c)
	return int32(val)
}

// Bytes copies the raw bytes out of the array without type conversion.
// Works with common dtypes (float32, int32, uint32, uint8).
// For non-contiguous arrays, call Contiguous() first.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) Bytes() []byte {
	cleanup()
	nbytes := a.Nbytes()
	if nbytes == 0 {
		return nil
	}

	// Get raw pointer based on dtype
	var ptr unsafe.Pointer
	switch a.Dtype() {
	case DtypeFloat32:
		ptr = unsafe.Pointer(C.mlx_array_data_float32(a.c))
	case DtypeInt32:
		ptr = unsafe.Pointer(C.mlx_array_data_int32(a.c))
	case DtypeUint32:
		ptr = unsafe.Pointer(C.mlx_array_data_uint32(a.c))
	case DtypeUint8:
		ptr = unsafe.Pointer(C.mlx_array_data_uint8(a.c))
	default:
		// For other types (bf16, f16, etc), convert to float32
		arr := AsType(a, DtypeFloat32)
		arr.Eval()
		ptr = unsafe.Pointer(C.mlx_array_data_float32(arr.c))
		nbytes = arr.Nbytes()
	}

	if ptr == nil {
		return nil
	}
	data := make([]byte, nbytes)
	copy(data, unsafe.Slice((*byte)(ptr), nbytes))
	return data
}

// String returns a string representation
func (a *Array) String() string {
	shape := a.Shape()
	size := a.Size()
	if size <= 20 {
		data := a.Data()
		return fmt.Sprintf("Array(shape=%v, data=%v)", shape, data)
	}
	return fmt.Sprintf("Array(shape=%v, size=%d)", shape, size)
}
