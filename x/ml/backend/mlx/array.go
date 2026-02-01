//go:build mlx

// Package mlx - Array-Grundstruktur und Datenkonvertierung
//
// Hauptfunktionen:
// - Array: MLX-Array Wrapper
// - newArray: Array mit Context-Tracking erstellen
// - FromFloats/FromInts: Tensor aus Go-Slices erstellen
// - Floats/Ints: Tensor-Daten als Go-Slices abrufen
// - Zeros/Empty: Leere Tensoren erstellen

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
static inline size_t stride(const mlx_array a, int i) {return mlx_array_strides(a)[i];}
*/
import "C"

import (
	"fmt"
	"log/slog"
	"runtime"
	"unsafe"

	"github.com/ollama/ollama/x/ml"
	"github.com/x448/float16"
)

// Array kapselt ein MLX-Array mit Kontext
type Array struct {
	name string
	a    C.mlx_array
	c    *Context

	sync func()
}

// newArray erstellt ein neues Array mit Context-Tracking
func newArray(ctx *Context, a C.mlx_array) *Array {
	// TODO measure impact and if this slows things down, make it conditional on some debugging flag at load time
	var name string
	_, f, l, ok := runtime.Caller(2)
	if ok {
		name = fmt.Sprintf("%s:%d", f, l)
	}

	t := &Array{
		name: name,
		a:    a,
		c:    ctx,
	}
	// DEBUG memory allocation problems...
	// slog.Info("XXX Allocated", "array", t, "a", a)
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	ctx.arrays = append(ctx.arrays, a)
	return t
}

// FromFloats erstellt einen Tensor aus float32-Slice
func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	u16s := make([]float16.Float16, len(s))
	for i := range u16s {
		u16s[i] = float16.Fromfloat32(s[i])
	}
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&u16s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_FLOAT16,
		),
	)
}

// Floats gibt Array-Daten als float32-Slice zurück
func (a *Array) Floats() []float32 {
	if a.sync != nil {
		a.sync()
	}
	l := (int)(C.mlx_array_size(a.a))

	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BFLOAT16:
		panic("bfloat16 not yet implemented")
	case C.MLX_FLOAT16:
		data := C.mlx_array_data_float16_asvoid(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		u16s := unsafe.Slice((*uint16)(data), l)
		f32s := make([]float32, len(u16s))
		for i := range u16s {
			f32s[i] = float16.Frombits(u16s[i]).Float32()
		}
		return f32s
	case C.MLX_FLOAT32:
		data := C.mlx_array_data_float32(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		f32s := unsafe.Slice((*float32)(data), l)
		return f32s
	default:
		panic(fmt.Sprintf("unsupported dtype for Floats: %d", C.mlx_array_dtype(a.a)))
	}
}

// FromInts erstellt einen Tensor aus int32-Slice
func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	)
}

// Ints gibt Array-Daten als int32-Slice zurück
func (a *Array) Ints() []int32 {
	if a.sync != nil {
		a.sync()
	}
	l := (int)(C.mlx_array_size(a.a))

	switch C.mlx_array_dtype(a.a) {
	case C.MLX_INT32:
		data := C.mlx_array_data_int32(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		i32s := unsafe.Slice((*int32)(data), l)
		return i32s

		// TODO other types via conversion?
	default:
		panic(fmt.Sprintf("unsupported dtype for Ints: %d", C.mlx_array_dtype(a.a)))
	}
}

// Zeros erstellt einen Null-Tensor
func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	sh := make([]C.int, len(shape))
	for i, s := range shape {
		sh[i] = (C.int)(s)
	}

	var r C.mlx_array
	C.mlx_zeros(
		&r,
		&sh[0],
		(C.size_t)(len(sh)),
		C.mlx_dtype(dtype),
		c.stream,
	)
	return newArray(c, r)
}

// Empty erstellt einen leeren Tensor (verwendet Zeros)
func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	// TODO more efficient impl?
	return c.Zeros(dtype, shape...)
}

// DType gibt den Datentyp zurück
func (a *Array) DType() ml.DType {
	return (ml.DType)(C.mlx_array_dtype(a.a))
}

// Dim gibt die Dimension an Index n zurück
func (a *Array) Dim(n int) int {
	return int(C.mlx_array_dim(a.a, C.int(n)))
}

// Stride gibt den Stride an Index n zurück
func (a *Array) Stride(n int) int {
	return (int)(C.stride(a.a, (C.int)(n)))
}

// Arange erstellt einen Bereichs-Tensor
func (c *Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	var r C.mlx_array
	C.mlx_arange(
		&r,
		C.double(start),
		C.double(stop),
		C.double(step),
		(C.mlx_dtype)(dtype),
		c.stream,
	)

	return newArray(c, r)
}

// Shape gibt die Form des Arrays zurück
func (a *Array) Shape() []int {
	shape := make([]int, C.mlx_array_ndim(a.a))
	for i := range shape {
		shape[i] = int(C.mlx_array_dim(a.a, C.int(i)))
	}

	return shape
}

// TypeString gibt den Datentyp als String zurück
func (a *Array) TypeString() string {
	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BOOL:
		return "bool"
	case C.MLX_UINT8:
		return "uint8"
	case C.MLX_UINT16:
		return "uint16"
	case C.MLX_UINT32:
		return "uint32"
	case C.MLX_UINT64:
		return "uint64"
	case C.MLX_INT8:
		return "int8"
	case C.MLX_INT16:
		return "int16"
	case C.MLX_INT32:
		return "int32"
	case C.MLX_INT64:
		return "int64"
	case C.MLX_FLOAT16:
		return "float16"
	case C.MLX_FLOAT32:
		return "float32"
	case C.MLX_BFLOAT16:
		return "bfloat16"
	case C.MLX_COMPLEX64:
		return "complex64"
	default:
		return "unknown"
	}
}

// ToString gibt eine String-Repräsentation zurück
func (a *Array) ToString() string {
	str := C.mlx_string_new()
	C.mlx_array_tostring(&str, a.a)
	s := C.mlx_string_data(str)
	defer C.mlx_string_free(str)
	return C.GoString(s)
}

// LogValue gibt einen slog.Value für Logging zurück
func (a *Array) LogValue() slog.Value {
	dims := int(C.mlx_array_ndim(a.a))
	strides := make([]int, dims)
	for i := range strides {
		strides[i] = int(C.stride(a.a, (C.int)(i)))
	}

	return slog.GroupValue(
		slog.String("name", a.name),
		slog.String("type", a.TypeString()),
		slog.Any("shape", a.Shape()),
		slog.Any("strides", strides),
		// slog.String("values", C.GoString(s)),
	)
}
