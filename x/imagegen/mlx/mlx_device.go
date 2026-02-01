//go:build mlx

// mlx_device.go - Device, Stream und Memory Management
//
// Enthaelt:
// - Device Control (SetDefaultDeviceGPU, SetDefaultDeviceCPU)
// - Metal/GPU Verfuegbarkeit
// - Stream Management
// - Memory Control (Cache, Limits)
// - GPU Trace Capture

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

static inline void set_default_stream(mlx_stream s) {
    _default_stream = s;
}
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// SetDefaultDeviceGPU sets the default device to GPU (Metal)
func SetDefaultDeviceGPU() {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
}

// SetDefaultDeviceCPU sets the default device to CPU
func SetDefaultDeviceCPU() {
	dev := C.mlx_device_new_type(C.MLX_CPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
}

// MetalIsAvailable returns true if Metal GPU is available
func MetalIsAvailable() bool {
	var available C._Bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}

// MetalStartCapture starts a GPU trace capture to the given file path.
// The path must not already exist. Run with MTL_CAPTURE_ENABLED=1 env var.
// Open the resulting .gputrace file in Xcode for analysis.
func MetalStartCapture(path string) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	C.mlx_metal_start_capture(cPath)
}

// MetalStopCapture stops the current GPU trace capture.
func MetalStopCapture() {
	C.mlx_metal_stop_capture()
}

// GPUIsAvailable returns true if any GPU (Metal or CUDA) is available
func GPUIsAvailable() bool {
	// On Linux with CUDA build, GPU is available
	// On macOS, check Metal availability
	if MetalIsAvailable() {
		return true
	}
	// CUDA is available if we compiled with CUDA support (Linux)
	return runtime.GOOS == "linux"
}

// GetDefaultDeviceType returns the current default device (0=CPU, 1=GPU)
func GetDefaultDeviceType() int {
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	var devType C.mlx_device_type
	C.mlx_device_get_type(&devType, dev)
	C.mlx_device_free(dev)
	return int(devType)
}

// Synchronize waits for all GPU operations to complete
func Synchronize() {
	C.mlx_synchronize(C.default_stream())
}

// Stream represents an MLX execution stream
type Stream struct {
	c C.mlx_stream
}

// NewStream creates a new execution stream on the default device
func NewStream() *Stream {
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	stream := C.mlx_stream_new_device(dev)
	C.mlx_device_free(dev)
	return &Stream{c: stream}
}

// Free releases the stream
func (s *Stream) Free() {
	if s.c.ctx != nil {
		C.mlx_stream_free(s.c)
		s.c.ctx = nil
	}
}

// SetDefaultStream sets the default stream for operations
func SetDefaultStream(s *Stream) {
	C.mlx_set_default_stream(s.c)
	C.set_default_stream(s.c) // Also update our cached stream
}

// GetDefaultStream returns the current default stream
func GetDefaultStream() *Stream {
	var stream C.mlx_stream
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	C.mlx_get_default_stream(&stream, dev)
	C.mlx_device_free(dev)
	return &Stream{c: stream}
}

// SynchronizeStream waits for all operations on the stream to complete
func SynchronizeStream(s *Stream) {
	C.mlx_synchronize(s.c)
}

// MetalGetCacheMemory returns the current cache memory usage in bytes
func MetalGetCacheMemory() uint64 {
	var size C.size_t
	C.mlx_get_cache_memory(&size)
	return uint64(size)
}

// MetalGetPeakMemory returns the peak memory usage in bytes
func MetalGetPeakMemory() uint64 {
	var size C.size_t
	C.mlx_get_peak_memory(&size)
	return uint64(size)
}

// MetalResetPeakMemory resets the peak memory counter
func MetalResetPeakMemory() {
	C.mlx_reset_peak_memory()
}

// MetalSetWiredLimit sets the wired memory limit and returns the previous limit
// This keeps tensors pinned in GPU memory for faster access
func MetalSetWiredLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_wired_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// MetalGetActiveMemory returns the current active memory usage in bytes
func MetalGetActiveMemory() uint64 {
	var size C.size_t
	C.mlx_get_active_memory(&size)
	return uint64(size)
}

// ClearCache clears the MLX memory cache
func ClearCache() {
	C.mlx_clear_cache()
}

// SetCacheLimit sets the free cache limit in bytes
// Setting to 0 disables caching (useful for memory-constrained generation)
// Returns the previous cache limit
func SetCacheLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// SetMemoryLimit sets the overall memory limit in bytes
// This is a guideline for maximum memory during graph evaluation.
// When Metal is available, defaults to 1.5x the max recommended working set.
// Returns the previous memory limit
func SetMemoryLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_memory_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// GetMemoryLimit returns the current memory limit in bytes
func GetMemoryLimit() uint64 {
	var size C.size_t
	C.mlx_get_memory_limit(&size)
	return uint64(size)
}

// EnableCompile enables global compilation/graph fusion
func EnableCompile() {
	C.mlx_enable_compile()
}

// DisableCompile disables global compilation
func DisableCompile() {
	C.mlx_disable_compile()
}

// SetCompileMode sets the compile mode
// 0=disabled, 1=no_simplify, 2=no_fuse, 3=enabled
func SetCompileMode(mode int) {
	C.mlx_set_compile_mode(C.mlx_compile_mode(mode))
}
