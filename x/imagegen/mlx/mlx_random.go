//go:build mlx

// mlx_random.go - Zufallszahlen und PRNG
//
// Enthaelt:
// - RandomKey, RandomSplit
// - RandomNormal, RandomUniform
// - RandomCategorical
// - RandN (einfache API)

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
import (
	"sync/atomic"
	"time"
)

// Global random seed counter for RandN
var randnSeedCounter uint64 = uint64(time.Now().UnixNano())

// RandomKey creates a PRNG key from a seed
func RandomKey(seed uint64) *Array {
	var res C.mlx_array
	C.mlx_random_key(&res, C.uint64_t(seed))
	return newArray(res)
}

// RandomSplit splits a PRNG key into two new keys
func RandomSplit(key *Array) (*Array, *Array) {
	var key1, key2 C.mlx_array
	C.mlx_random_split(&key1, &key2, key.c, C.default_stream())
	return newArray(key1), newArray(key2)
}

// RandomCategoricalWithKey samples from categorical distribution using provided key.
func RandomCategoricalWithKey(logits, key *Array, axis int, numSamples int) *Array {
	res := C.mlx_array_new()
	C.mlx_random_categorical_num_samples(&res, logits.c, C.int(axis), C.int(numSamples), key.c, C.default_stream())
	return newArray(res)
}

// RandomCategorical samples using global RandomState.
// For simple scripts - production code should use RandomCategoricalWithKey with explicit key management.
func RandomCategorical(logits *Array, axis int, numSamples int) *Array {
	randomStateMu.Lock()
	oldKey := RandomState[0]
	key1, key2 := RandomSplit(oldKey)
	Keep(key1) // key1 becomes the new global state
	oldKey.Free()
	RandomState[0] = key1
	randomStateMu.Unlock()
	return RandomCategoricalWithKey(logits, key2, axis, numSamples)
}

// RandomNormal creates a random normal (Gaussian) tensor in float32
func RandomNormal(shape []int32, seed uint64) *Array {
	return RandomNormalWithDtype(shape, seed, DtypeFloat32)
}

// RandomNormalWithDtype creates a random normal (Gaussian) tensor with specified dtype
func RandomNormalWithDtype(shape []int32, seed uint64, dtype Dtype) *Array {
	key := RandomKey(seed)
	res := C.mlx_array_new()
	C.mlx_random_normal(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dtype), 0.0, 1.0, key.c, C.default_stream())
	return newArray(res)
}

// RandomUniform generates uniform random values in [0, 1) with the given shape
func RandomUniform(shape []int32, seed uint64) *Array {
	key := RandomKey(seed)
	low := C.mlx_array_new_float(0.0)
	high := C.mlx_array_new_float(1.0)
	res := C.mlx_array_new()
	C.mlx_random_uniform(&res, low, high, int32ToCInt(shape), C.size_t(len(shape)), C.MLX_FLOAT32, key.c, C.default_stream())
	C.mlx_array_free(low)
	C.mlx_array_free(high)
	return newArray(res)
}

// RandN creates an array of random samples from a standard normal distribution
func RandN(shape []int32) *Array {
	// Use incrementing seed for unique random values each call
	seed := atomic.AddUint64(&randnSeedCounter, 1)
	return RandomNormal(shape, seed)
}
