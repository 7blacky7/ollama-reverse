//go:build mlx

// mlx_init.go - MLX Initialisierung
//
// Enthaelt:
// - InitMLX, IsMLXAvailable, GetMLXInitError
// - init() fuer automatische Initialisierung

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"time"
)

// InitMLX initializes the MLX library by dynamically loading libmlxc.
// This must be called before using any MLX functions.
// Returns an error if the library cannot be loaded.
func InitMLX() error {
	if mlxInitialized {
		return mlxInitError
	}

	// Try to load the MLX dynamic library
	ret := C.mlx_dynamic_init()
	if ret != 0 {
		errMsg := C.GoString(C.mlx_dynamic_error())
		mlxInitError = fmt.Errorf("failed to initialize MLX: %s", errMsg)
		return mlxInitError
	}

	// Initialize all function pointers via dlsym
	handle := C.mlx_get_handle()
	ret = C.mlx_load_functions(handle)
	if ret != 0 {
		mlxInitError = fmt.Errorf("failed to load MLX function symbols")
		return mlxInitError
	}

	mlxInitialized = true
	mlxInitError = nil
	return nil
}

// IsMLXAvailable returns whether MLX was successfully initialized
func IsMLXAvailable() bool {
	return mlxInitialized && mlxInitError == nil
}

// GetMLXInitError returns any error that occurred during MLX initialization
func GetMLXInitError() error {
	return mlxInitError
}

func init() {
	// Initialize MLX dynamic library first
	if err := InitMLX(); err != nil {
		// Don't panic in init - let the caller handle the error
		// Store the error for later retrieval
		mlxInitError = err
		return
	}

	// Lock main goroutine to OS thread for CUDA context stability.
	// CUDA contexts are bound to threads; Go can migrate goroutines between threads.
	runtime.LockOSThread()
	RandomState[0] = RandomKey(uint64(time.Now().UnixMilli()))
	Keep(RandomState[0]) // Global state should persist
}
