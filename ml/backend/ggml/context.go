// context.go - Context-Struktur und Methoden
// Enthält: Context struct, Forward(), Compute(), Reserve(), Tensor-Erstellung

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// Context repräsentiert einen GGML-Berechnungskontext
type Context struct {
	b *Backend

	ctx   *C.struct_ggml_context
	graph *C.struct_ggml_cgraph

	// batchSize ist ein Hinweis zur Optimierung
	batchSize int

	// buft ist der Buffer-Typ für neue Tensoren
	buft C.ggml_backend_buffer_type_t

	// allocatedBuffers sind Buffer für Tensoren in diesem Kontext
	allocatedBuffers *[]C.ggml_backend_buffer_t

	// maxGraphNodes ist die maximale Anzahl an Graph-Knoten
	maxGraphNodes int

	// layer ist der Graph-Layer für diesen Kontext (für Cache)
	layer int
}

// Input gibt einen Kontext für Eingabe-Tensoren zurück
func (c *Context) Input() ml.Context {
	if c.b.input != nil {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             c.b.input,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
			layer:            -1,
		}
	}

	return c
}

// Layer gibt einen Kontext für einen bestimmten Layer zurück
func (c *Context) Layer(i int) ml.Context {
	if layer, ok := c.b.layers[i]; ok {
		return &Context{
			b:                c.b,
			ctx:              c.ctx,
			buft:             layer.bt,
			allocatedBuffers: c.allocatedBuffers,
			maxGraphNodes:    c.maxGraphNodes,
			layer:            i,
		}
	}

	return c
}

// Forward fügt Tensoren zum Berechnungsgraphen hinzu
func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	if c.graph == nil {
		c.graph = C.ggml_new_graph_custom(c.ctx, C.size_t(c.maxGraphNodes), false)
	}

	for _, tensor := range tensors {
		C.ggml_build_forward_expand(c.graph, tensor.(*Tensor).t)
	}

	return c
}

// SetBatchSize setzt die Batch-Größe für Optimierung
func (c *Context) SetBatchSize(batchSize int) {
	c.batchSize = batchSize
}

// Compute führt die Berechnung des Graphen aus
func (c *Context) Compute(tensors ...ml.Tensor) {
	c.ComputeWithNotify(nil, tensors...)
}

// ComputeWithNotify führt die Berechnung mit optionalem Callback aus
func (c *Context) ComputeWithNotify(cb func(), tensors ...ml.Tensor) {
	c.b.schedMu.Lock()
	defer c.b.schedMu.Unlock()
	if cb != nil {
		go cb()
	}

	if c.batchSize > 0 {
		C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
	}

	if status := C.ggml_backend_sched_graph_compute_async(c.b.sched, c.graph); status != C.GGML_STATUS_SUCCESS {
		panic(fmt.Errorf("error computing ggml graph: %v", status))
	}
	C.ggml_backend_sched_reset(c.b.sched)

	needSync := true
	sync := func() {
		if needSync {
			C.ggml_backend_sched_synchronize(c.b.sched)
			needSync = false
		}
	}

	for _, t := range tensors {
		if C.ggml_nbytes(t.(*Tensor).t) > 0 {
			t.(*Tensor).sync = sync
		}
	}
}

// Reserve reserviert Speicher für den Graphen
func (c *Context) Reserve() {
	if c.batchSize > 0 {
		C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
	}

	reserved := C.ggml_backend_sched_reserve(c.b.sched, c.graph)

	logutil.Trace("compute graph", "nodes", C.ggml_graph_n_nodes(c.graph), "splits", C.ggml_backend_sched_get_n_splits(c.b.sched))

	// Reserve kann mehrfach aufgerufen werden - wir wollen den letzten Lauf
	for _, bt := range c.b.schedBufts {
		c.b.btDeviceMemory[bt].Graph = 0
	}

	for i := range c.b.schedBackends {
		bufferSize := C.ggml_backend_sched_get_attempted_buffer_size(c.b.sched, c.b.schedBackends[i])
		c.b.btDeviceMemory[c.b.schedBufts[i]].Graph += uint64(bufferSize)

		logutil.Trace("compute graph", "backend", C.GoString(C.ggml_backend_name(c.b.schedBackends[i])),
			"buffer_type", C.GoString(C.ggml_backend_buft_name(c.b.schedBufts[i])), "size", format.HumanBytes2(uint64(bufferSize)))
	}

	if !reserved {
		panic(ml.ErrNoMem{BackendMemory: *c.b.requiredMemory})
	}
}

// MaxGraphNodes gibt die maximale Anzahl an Graph-Knoten zurück
func (c *Context) MaxGraphNodes() int {
	return c.maxGraphNodes
}

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

// checkShape prüft ob die Daten zur Shape passen
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

// Close gibt die Ressourcen des Kontexts frei
func (c *Context) Close() {
	if c != nil {
		for _, b := range *c.allocatedBuffers {
			C.ggml_backend_buffer_free(b)
		}
		*c.allocatedBuffers = nil

		C.ggml_free(c.ctx)
	}
}

// shapeToGGML konvertiert eine Go-Shape in GGML-Format
func shapeToGGML(shape []int) *C.int64_t {
	sh := make([]C.int64_t, len(shape))
	for i, s := range shape {
		sh[i] = C.int64_t(s)
	}

	return &sh[0]
}

// ggmlDType konvertiert ml.DType zu GGML-Typ
func ggmlDType(dtype ml.DType) uint32 {
	switch dtype {
	case ml.DTypeF32:
		return C.GGML_TYPE_F32
	case ml.DTypeF16:
		return C.GGML_TYPE_F16
	case ml.DTypeQ80:
		return C.GGML_TYPE_Q8_0
	case ml.DTypeQ40:
		return C.GGML_TYPE_Q4_0
	case ml.DTypeI32:
		return C.GGML_TYPE_I32
	case ml.DTypeMXFP4:
		return C.GGML_TYPE_MXFP4
	default:
		panic("unsupported dtype")
	}
}

// inferShape berechnet automatisch eine -1 Dimension
func inferShape(t *Tensor, shape []int) {
	total := 1
	for _, dim := range t.Shape() {
		total *= dim
	}

	dim := -1
	for i := range shape {
		switch shape[i] {
		case -1:
			if dim != -1 {
				panic("only one dimension can be inferred")
			}
			dim = i
		case 0:
			panic("dimension cannot be zero")
		default:
			if total%shape[i] != 0 {
				panic("cannot infer dimension")
			}

			total /= shape[i]
		}
	}

	if dim != -1 {
		shape[dim] = total
	}
}
