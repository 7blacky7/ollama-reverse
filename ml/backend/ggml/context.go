// context.go - Context-Struktur und Kern-Methoden
// Enthaelt: Context struct, Input(), Layer(), Forward(), Compute(), Reserve(), Close()

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"fmt"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// Context repraesentiert einen GGML-Berechnungskontext
type Context struct {
	b *Backend

	ctx   *C.struct_ggml_context
	graph *C.struct_ggml_cgraph

	// batchSize ist ein Hinweis zur Optimierung
	batchSize int

	// buft ist der Buffer-Typ fuer neue Tensoren
	buft C.ggml_backend_buffer_type_t

	// allocatedBuffers sind Buffer fuer Tensoren in diesem Kontext
	allocatedBuffers *[]C.ggml_backend_buffer_t

	// maxGraphNodes ist die maximale Anzahl an Graph-Knoten
	maxGraphNodes int

	// layer ist der Graph-Layer fuer diesen Kontext (fuer Cache)
	layer int
}

// Input gibt einen Kontext fuer Eingabe-Tensoren zurueck
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

// Layer gibt einen Kontext fuer einen bestimmten Layer zurueck
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

// Forward fuegt Tensoren zum Berechnungsgraphen hinzu
func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	if c.graph == nil {
		c.graph = C.ggml_new_graph_custom(c.ctx, C.size_t(c.maxGraphNodes), false)
	}

	for _, tensor := range tensors {
		C.ggml_build_forward_expand(c.graph, tensor.(*Tensor).t)
	}

	return c
}

// SetBatchSize setzt die Batch-Groesse fuer Optimierung
func (c *Context) SetBatchSize(batchSize int) {
	c.batchSize = batchSize
}

// Compute fuehrt die Berechnung des Graphen aus
func (c *Context) Compute(tensors ...ml.Tensor) {
	c.ComputeWithNotify(nil, tensors...)
}

// ComputeWithNotify fuehrt die Berechnung mit optionalem Callback aus
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

// Reserve reserviert Speicher fuer den Graphen
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

// MaxGraphNodes gibt die maximale Anzahl an Graph-Knoten zurueck
func (c *Context) MaxGraphNodes() int {
	return c.maxGraphNodes
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
