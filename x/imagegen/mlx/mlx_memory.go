//go:build mlx

// mlx_memory.go - Memory Management fuer MLX Arrays
//
// Enthaelt:
// - Array Lifecycle (newArray, Free, Keep)
// - Eval und Cleanup Funktionen
// - Collect und FreeStruct fuer Strukturen

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
import "reflect"

func newArray(array C.mlx_array) *Array {
	if InClosureCallback() {
		return &Array{c: array}
	}
	a := arrayPool.Get().(*Array)
	a.c = array
	a.freed = false
	a.kept = false
	arrays = append(arrays, a)
	return a
}

// Collect uses reflection to find all *Array fields in a struct (recursively).
func Collect(v any) []*Array {
	var arrs []*Array
	seen := make(map[uintptr]bool)
	collect(reflect.ValueOf(v), &arrs, seen)
	return arrs
}

func collect(v reflect.Value, arrs *[]*Array, seen map[uintptr]bool) {
	if !v.IsValid() {
		return
	}
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return
		}
		ptr := v.Pointer()
		if seen[ptr] {
			return
		}
		seen[ptr] = true
		if arr, ok := v.Interface().(*Array); ok {
			if arr != nil && arr.c.ctx != nil {
				*arrs = append(*arrs, arr)
			}
			return
		}
		collect(v.Elem(), arrs, seen)
		return
	}
	if v.Kind() == reflect.Struct {
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			if field.CanInterface() {
				collect(field, arrs, seen)
			}
		}
		return
	}
	if v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			collect(v.Index(i), arrs, seen)
		}
		return
	}
	if v.Kind() == reflect.Map {
		for _, key := range v.MapKeys() {
			collect(v.MapIndex(key), arrs, seen)
		}
		return
	}
	if v.Kind() == reflect.Interface && !v.IsNil() {
		collect(v.Elem(), arrs, seen)
	}
}

// FreeStruct releases all *Array fields in a struct (recursively).
func FreeStruct(v any) {
	for _, arr := range Collect(v) {
		arr.Free()
	}
}

// Keep marks arrays to persist across Eval() cleanup.
func Keep(arrs ...*Array) {
	for _, a := range arrs {
		if a != nil {
			a.kept = true
		}
	}
}

// cleanup frees non-kept arrays and compacts the live array list.
func cleanup() int {
	freed := 0
	n := 0
	for _, a := range arrays {
		if a.kept {
			arrays[n] = a
			n++
		} else if a.c.ctx != nil && !a.freed {
			C.mlx_array_free(a.c)
			a.c.ctx = nil
			arrayPool.Put(a)
			freed++
		}
	}
	arrays = arrays[:n]
	return freed
}

// Eval synchronously evaluates arrays and cleans up non-kept arrays.
func Eval(outputs ...*Array) []*Array {
	for _, o := range outputs {
		if o != nil {
			o.kept = true
		}
	}
	cleanup()
	if len(outputs) > 0 {
		evalHandles = evalHandles[:0]
		for _, o := range outputs {
			if o != nil {
				evalHandles = append(evalHandles, o.c)
			}
		}
		if len(evalHandles) > 0 {
			vec := C.mlx_vector_array_new_data(&evalHandles[0], C.size_t(len(evalHandles)))
			C.mlx_eval(vec)
			C.mlx_vector_array_free(vec)
		}
	}
	return outputs
}

// AsyncEval dispatches async evaluation and cleans up non-kept arrays.
func AsyncEval(outputs ...*Array) {
	for _, o := range outputs {
		if o != nil {
			o.kept = true
		}
	}
	cleanup()
	if len(outputs) > 0 {
		evalHandles = evalHandles[:0]
		for _, o := range outputs {
			if o != nil {
				evalHandles = append(evalHandles, o.c)
			}
		}
		if len(evalHandles) > 0 {
			vec := C.mlx_vector_array_new_data(&evalHandles[0], C.size_t(len(evalHandles)))
			C.mlx_async_eval(vec)
			C.mlx_vector_array_free(vec)
		}
	}
}

// Sync waits for all async operations to complete.
func Sync() { C.mlx_synchronize(C.default_stream()) }

// Free marks this array for cleanup on the next Eval().
func (a *Array) Free() {
	if a != nil {
		a.kept = false
	}
}

// Eval evaluates this single array and runs cleanup.
func (a *Array) Eval() *Array {
	Eval(a)
	return a
}

// Valid returns true if the array hasn't been freed.
func (a *Array) Valid() bool { return a != nil && a.c.ctx != nil }

// Kept returns true if the array is marked to survive Eval() cleanup.
func (a *Array) Kept() bool { return a != nil && a.kept }
