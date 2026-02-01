// context_tensors.go - Tensor-Erstellungsmethoden fuer Context
// Enthaelt: newTensor(), Empty(), Zeros(), FromBytes(), FromFloats(), FromInts(), Arange()

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"fmt"

	"github.com/ollama/ollama/ml"
)

// newTensor erstellt einen neuen Tensor mit dem aktuellen Buffer-Typ
func (c *Context) newTensor(dtype ml.DType, shape []int) *Tensor {
	if c.buft == nil {
		panic("set Input or Layer before creating tensors")
	}

	cdtype := ggmlDType(dtype)

	if len(shape) < 1 || shape[0] == 0 {
		var shape C.int64_t = 0
		return &Tensor{b: c.b, t: C.ggml_new_tensor(c.ctx, cdtype, 1, &shape)}
	} else if len(shape) > 4 {
		panic("unsupported number of dimensions")
	}

	for _, dim := range shape {
		if dim < 1 {
			panic("invalid shape")
		}
	}

	t := C.ggml_new_tensor(c.ctx, cdtype, C.int(len(shape)), shapeToGGML(shape))
	size := pad(C.ggml_backend_buft_get_alloc_size(c.buft, t), C.ggml_backend_buft_get_alignment(c.buft))

	b := C.ggml_backend_buft_alloc_buffer(c.buft, size)
	if c.layer >= 0 {
		c.b.btDeviceMemory[c.buft].Cache[c.layer] += uint64(size)
	}

	if b == nil {
		panic(ml.ErrNoMem{BackendMemory: *c.b.requiredMemory})
	}

	*c.allocatedBuffers = append(*c.allocatedBuffers, b)
	C.ggml_backend_tensor_alloc(b, t, C.ggml_backend_buffer_get_base(b))
	return &Tensor{b: c.b, t: t}
}

// Empty erstellt einen leeren Tensor
func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	return c.newTensor(dtype, shape)
}

// Zeros erstellt einen mit Nullen initialisierten Tensor
func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	t := c.newTensor(dtype, shape)
	if c.b.allocMemory {
		C.ggml_set_zero(t.t)
	}
	return t
}

// checkShape prueft ob die Daten zur Shape passen
func checkShape[S ~[]E, E any](s S, shape ...int) {
	n := len(s)

	if n == 0 {
		return
	}

	for _, v := range shape {
		n /= v
	}

	if n != 1 {
		panic(fmt.Errorf("invalid shape: %v", shape))
	}
}

// FromBytes erstellt einen Tensor aus Bytes
func (c Context) FromBytes(dtype ml.DType, s []uint8, shape ...int) ml.Tensor {
	t := c.newTensor(dtype, shape)
	if c.b.allocMemory {
		t.FromBytes(s)
	}

	return t
}

// FromFloats erstellt einen Tensor aus Float32-Werten
func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	checkShape(s, shape...)

	t := c.newTensor(ml.DTypeF32, shape)

	if c.b.allocMemory {
		t.FromFloats(s)
	}

	return t
}

// FromInts erstellt einen Tensor aus Int32-Werten
func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	checkShape(s, shape...)

	t := c.newTensor(ml.DTypeI32, shape)
	if c.b.allocMemory {
		t.FromInts(s)
	}

	return t
}

// Arange erstellt einen Tensor mit aufsteigenden Werten
func (c Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	switch dtype {
	case ml.DTypeF32:
		return &Tensor{
			b: c.b,
			t: C.ggml_arange(c.ctx, C.float(start), C.float(stop), C.float(step)),
		}
	case ml.DTypeI32:
		arange := make([]int32, 0, int((stop-start)/step))
		for i := start; i < stop; i += step {
			arange = append(arange, int32(i))
		}

		return c.Input().FromInts(arange, len(arange))
	default:
		panic("unsupported dtype for arange")
	}
}
