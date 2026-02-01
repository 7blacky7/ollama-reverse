//go:build mlx

// mlx_io.go - Datei I/O fuer MLX
//
// Enthaelt:
// - Safetensors Loading/Saving
// - NPY Loading
// - SafetensorsFile Struct und Methoden

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>

static mlx_stream _cpu_stream = {0};

// CPU stream for file loading (Load primitive only runs on CPU)
static inline mlx_stream cpu_stream() {
    if (_cpu_stream.ctx == NULL) {
        _cpu_stream = mlx_default_cpu_stream_new();
    }
    return _cpu_stream;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// SafetensorsFile represents a loaded safetensors file
type SafetensorsFile struct {
	arrays   C.mlx_map_string_to_array
	metadata C.mlx_map_string_to_string
}

// LoadSafetensorsNative loads a safetensors file using MLX's optimized loader
// Note: Uses CPU stream because Load primitive only runs on CPU
func LoadSafetensorsNative(path string) (*SafetensorsFile, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var arrays C.mlx_map_string_to_array
	var metadata C.mlx_map_string_to_string
	if C.mlx_load_safetensors(&arrays, &metadata, cPath, C.cpu_stream()) != 0 {
		return nil, fmt.Errorf("failed to load safetensors: %s", path)
	}
	return &SafetensorsFile{arrays: arrays, metadata: metadata}, nil
}

// Get retrieves a tensor by name
func (s *SafetensorsFile) Get(name string) *Array {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var arr C.mlx_array
	if C.mlx_map_string_to_array_get(&arr, s.arrays, cName) != 0 {
		return nil
	}
	if arr.ctx == nil {
		return nil
	}
	return newArray(arr)
}

// Set replaces a tensor in the map (like Python's weights[k] = v)
func (s *SafetensorsFile) Set(name string, arr *Array) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_map_string_to_array_insert(s.arrays, cName, arr.c)
}

// Count returns the number of tensors (not directly available, would need iterator)
func (s *SafetensorsFile) Count() int {
	// mlx-c doesn't have a direct count - would need to iterate
	return 0
}

// Free releases the safetensors file
func (s *SafetensorsFile) Free() {
	C.mlx_map_string_to_array_free(s.arrays)
	C.mlx_map_string_to_string_free(s.metadata)
}

// SaveSafetensors saves arrays to a safetensors file using MLX's native implementation.
// This correctly handles all dtypes including uint32 for quantized weights.
func SaveSafetensors(path string, arrs map[string]*Array) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	// Create the map
	cArrays := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(cArrays)

	// Add each array to the map
	for name, arr := range arrs {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(cArrays, cName, arr.c)
		C.free(unsafe.Pointer(cName))
	}

	// Create empty metadata (optional)
	cMeta := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(cMeta)

	// Save
	if C.mlx_save_safetensors(cPath, cArrays, cMeta) != 0 {
		return fmt.Errorf("failed to save safetensors: %s", path)
	}
	return nil
}

// LoadNpy loads a numpy array from an npy file
// Note: Uses CPU stream because Load primitive only runs on CPU
func LoadNpy(path string) (*Array, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var arr C.mlx_array
	if C.mlx_load(&arr, cPath, C.cpu_stream()) != 0 {
		return nil, fmt.Errorf("failed to load npy: %s", path)
	}
	if arr.ctx == nil {
		return nil, fmt.Errorf("failed to load npy: %s", path)
	}
	return newArray(arr), nil
}
