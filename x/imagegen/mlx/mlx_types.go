//go:build mlx

// mlx_types.go - Grundlegende Typen und Konstanten fuer MLX
//
// Enthaelt:
// - Dtype Definition und String()
// - Array Struct Definition
// - Globale Variablen fuer Array-Tracking

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
*/
import "C"
import "sync"

// Dtype represents MLX data types
type Dtype int

const (
	DtypeBool      Dtype = C.MLX_BOOL
	DtypeUint8     Dtype = C.MLX_UINT8
	DtypeUint16    Dtype = C.MLX_UINT16
	DtypeUint32    Dtype = C.MLX_UINT32
	DtypeUint64    Dtype = C.MLX_UINT64
	DtypeInt8      Dtype = C.MLX_INT8
	DtypeInt16     Dtype = C.MLX_INT16
	DtypeInt32     Dtype = C.MLX_INT32
	DtypeInt64     Dtype = C.MLX_INT64
	DtypeFloat16   Dtype = C.MLX_FLOAT16
	DtypeFloat32   Dtype = C.MLX_FLOAT32
	DtypeFloat64   Dtype = C.MLX_FLOAT64
	DtypeBFloat16  Dtype = C.MLX_BFLOAT16
	DtypeComplex64 Dtype = C.MLX_COMPLEX64
)

// String implements fmt.Stringer for Dtype
func (d Dtype) String() string {
	switch d {
	case DtypeBool:
		return "bool"
	case DtypeUint8:
		return "u8"
	case DtypeUint16:
		return "u16"
	case DtypeUint32:
		return "u32"
	case DtypeUint64:
		return "u64"
	case DtypeInt8:
		return "i8"
	case DtypeInt16:
		return "i16"
	case DtypeInt32:
		return "i32"
	case DtypeInt64:
		return "i64"
	case DtypeFloat16:
		return "f16"
	case DtypeFloat32:
		return "f32"
	case DtypeFloat64:
		return "f64"
	case DtypeBFloat16:
		return "bf16"
	case DtypeComplex64:
		return "c64"
	default:
		return "unknown"
	}
}

// ItemSize returns the size in bytes of one element for this dtype
func (d Dtype) ItemSize() int64 {
	switch d {
	case DtypeBool, DtypeUint8, DtypeInt8:
		return 1
	case DtypeUint16, DtypeInt16, DtypeFloat16, DtypeBFloat16:
		return 2
	case DtypeUint32, DtypeInt32, DtypeFloat32:
		return 4
	case DtypeUint64, DtypeInt64, DtypeFloat64, DtypeComplex64:
		return 8
	default:
		return 4
	}
}

// Array wraps an MLX array handle.
// Arrays are freed via Eval() cleanup (deterministic) or GC (fallback).
type Array struct {
	c     C.mlx_array
	freed bool // Prevents double-free
	kept  bool // If true, survives Eval() cleanup
}

// arrays tracks all live arrays. On Eval(), non-kept arrays are freed.
// Not goroutine-safe.
var arrays = make([]*Array, 0, 4096)

// evalHandles is a pre-allocated slice for passing arrays to MLX eval.
var evalHandles = make([]C.mlx_array, 0, 64)

// arrayPool reduces allocations for intermediate arrays
var arrayPool = sync.Pool{
	New: func() any { return &Array{} },
}

// RandomState is the global PRNG state, analogous to mx.random.state in Python.
var RandomState = []*Array{nil}
var randomStateMu sync.Mutex

// MLX initialization state
var mlxInitialized bool
var mlxInitError error
