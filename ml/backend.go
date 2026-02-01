// backend.go - Backend-Interface und Registrierung fuer ML-Modelle
// Dieses Modul definiert das Backend-Interface und die Backend-Factory-Funktionen.
package ml

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/fs"
)

// Backend represents a model execution backend (e.g., GGML).
type Backend interface {
	// Close frees all memory associated with this backend
	Close()

	Load(ctx context.Context, progress func(float32)) error

	// BackendMemory returns the memory allocations that were made for this model
	BackendMemory() BackendMemory

	Config() fs.Config
	Get(name string) Tensor
	NewContext() Context
	NewContextSize(size int) Context

	// Enumerate the devices available for inference via this backend
	BackendDevices() []DeviceInfo
}

// BackendCacheConfig should be implemented by backends that need special output
// from the cache to meet specific requirements. It is frequently implemented in
// conjunction with ScaledDotProductAttention.
type BackendCacheConfig interface {
	CacheConfig() CacheConfig
}

// CacheConfig controls optimizations (mostly backend-specific) that may transform
// the output the cache to work better with specific kernels.
type CacheConfig struct {
	// CachePadding specifies the multiple for the number of tokens of cache history
	// that will be returned from cache Get for k, v and mask. The capacity of the
	// cache itself will also be increased to a multiple of this size if needed.
	CachePadding int

	// PermutedV performs Permute(ctx, 1, 2, 0, 3) on v tensors stored via Put
	// and return the permuted version via Get. This uses the cache copy operation
	// to avoid a Contiguous call on the permuted tensor.
	PermutedV bool

	// MaskDType specifies the data type for generating the mask. If unset it will
	// default to DTypeF32.
	MaskDType DType
}

// BackendParams controls how the backend loads and executes models
type BackendParams struct {
	// AllocMemory causes the backend to allocate memory for the model. If
	// false, this is only being used for discovering the required amount of
	// memory and cannot load the model for running.
	AllocMemory bool

	// NumThreads sets the number of threads to use if running on the CPU
	NumThreads int

	// GPULayers is the set of layers to offload to GPUs
	GPULayers GPULayersList

	// FlashAttention indicates that we should use a fused flash attention kernel
	FlashAttention FlashAttentionType
}

var backends = make(map[string]func(string, BackendParams) (Backend, error))

// RegisterBackend registers a backend factory function.
func RegisterBackend(name string, f func(string, BackendParams) (Backend, error)) {
	if _, ok := backends[name]; ok {
		panic("backend: backend already registered")
	}

	backends[name] = f
}

// NewBackend creates a new backend instance for the given model path.
func NewBackend(modelPath string, params BackendParams) (Backend, error) {
	if backend, ok := backends["ggml"]; ok {
		return backend(modelPath, params)
	}

	return nil, fmt.Errorf("unsupported backend")
}
