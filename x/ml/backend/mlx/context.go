//go:build mlx

// Package mlx - Context-Verwaltung und Tensor-Vergleich
//
// Hauptfunktionen:
// - Context: GPU/CPU Stream-Verwaltung
// - Compute/Forward: Asynchrone Tensor-Evaluierung
// - CompareWith: Tensor-Vergleich mit Safetensor-Dateien
// - Similarity-Funktionen: Cosine, Euclidean, Manhattan

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/x/ml"
)

// Context verwaltet MLX-Streams und Arrays
type Context struct {
	stream C.mlx_stream

	mu     sync.Mutex
	arrays []C.mlx_array // TODO should we do some bookkeeping to ensure none of these Arrays are still lingering?
}

// Close gibt alle Ressourcen frei
func (c *Context) Close() {
	// C.mlx_synchronize(c.stream) // ???
	C.mlx_stream_free(c.stream)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, a := range c.arrays {
		slog.Info("XXX freeing", "array", a)
		C.mlx_array_free(a)
	}
}

// Compute evaluiert Tensoren asynchron
func (c *Context) Compute(tensors ...ml.Tensor) {
	// TODO - for the zero tensor case this feels like it might not be correct...
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range tensors {
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
		t.(*Array).sync = sync
	}
	C.mlx_async_eval(vec)
}

// Forward evaluiert Tensoren und gibt den Context zurück
func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	for _, t := range tensors {
		t.(*Array).sync = sync
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
	}
	C.mlx_async_eval(vec)
	return c
}

// Input gibt den Context zurück
func (c *Context) Input() ml.Context {
	return c
}

// Layer gibt den Context zurück (Layer-unabhängig)
func (c *Context) Layer(_ int) ml.Context {
	return c
}

// RandomNormal erzeugt zufällige Normalverteilung
func (c *Context) RandomNormal(shape []int, dtype ml.DType, loc, scale float32, key ml.Tensor) ml.Tensor {
	var r C.mlx_array
	var k C.mlx_array
	if key != nil {
		k = key.(*Array).a
	}
	sh := make([]C.int, len(shape))
	for i := range shape {
		sh[i] = C.int(shape[i])
	}
	C.mlx_random_normal(
		&r,
		&sh[0],
		C.size_t(len(shape)),
		C.mlx_dtype(dtype),
		C.float(loc),
		C.float(scale),
		k,
		c.stream,
	)
	return newArray(c, r)
}

// CompareWith vergleicht Tensoren mit Datei-Tensoren
func (c *Context) CompareWith(filepath string, tensors map[string]ml.Tensor, abortOnError bool) (err error) {
	const minCosine = float32(0.96) // TODO too low...
	fileTensors := map[string]*Array{}
	defer func() {
		if err != nil {
			for k, v := range tensors {
				fmt.Fprintln(os.Stderr, "input tensor "+k+"\n"+v.ToString())
				if fv, ok := fileTensors[k]; ok {
					fmt.Fprintln(os.Stderr, " file tensor "+k+"\n"+fv.ToString())
				} else {
					fmt.Fprintln(os.Stderr, " file tensor "+k+" missing!\n")
				}
			}
		}
		if abortOnError {
			if err != nil {
				panic(fmt.Sprintf("%s", err))
			}
		}
	}()
	if _, err = os.Stat(filepath); err != nil {
		filepath += ".safetensors"
		if _, err = os.Stat(filepath); err != nil {
			err = fmt.Errorf("failed to stat %s: %w", filepath, err)
			return
		}
		err = nil
	}
	// slog.Info("Loading tensors from", "filename", filepath)
	cFilename := C.CString(filepath)
	defer C.free(unsafe.Pointer(cFilename))
	data := C.mlx_map_string_to_array_new() // TODO is this needed or just var it?
	metadata := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_array_free(data)
	defer C.mlx_map_string_to_string_free(metadata)

	stream := C.mlx_default_cpu_stream_new()

	if C.mlx_load_safetensors(&data, &metadata, cFilename, stream) != 0 {
		// TODO with the current error handling, this will never happen
		err = fmt.Errorf("load failed")
		return
	}

	it := C.mlx_map_string_to_array_iterator_new(data)
	allTensors := []ml.Tensor{}
	for _, t := range tensors {
		allTensors = append(allTensors, t)
	}

	for {
		var key *C.cchar_t
		var value C.mlx_array
		defer C.mlx_array_free(value)
		if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
			break
		}
		k := C.GoString((*C.char)(key))
		var r C.mlx_array
		defer C.mlx_array_free(r)
		C.mlx_astype(
			&r,
			value,
			C.MLX_FLOAT32,
			stream,
		)

		fileTensors[k] = &Array{
			name: k,
			a:    r,
		}
		// slog.Info("XXX read", "tensor", t, "type", t.TypeString())
		allTensors = append(allTensors, fileTensors[k])
	}
	c.Forward(allTensors...)
	for k, t := range tensors {
		a, ok := fileTensors[k]
		if !ok {
			err = fmt.Errorf("tensor named %s not found in file", k)
			return
		}
		if !reflect.DeepEqual(a.Shape(), t.Shape()) {
			err = fmt.Errorf("mismatched shapes:  file: %v vs. input %v", a.Shape(), t.Shape())
			return
		}
		// slog.Info("XXX shapes match", "shape", t.Shape())
		// TODO handle int types...
		tDType := t.DType()
		if tDType != ml.DTypeFloat16 && tDType != ml.DTypeFloat32 {
			var r C.mlx_array
			defer C.mlx_array_free(r)
			C.mlx_astype(
				&r,
				t.(*Array).a,
				C.MLX_FLOAT32,
				stream,
			)
			t = &Array{
				a: r,
			}
			c.Forward(t)
		}

		af := a.Floats()
		tf := t.Floats()
		cos := cosineSimilarity(af, tf)
		diff := a.Sub(c, t)
		min := diff.Min(c, nil, true)
		max := diff.Max(c, nil, true)
		c.Forward(min, max)
		minf := min.Floats()
		maxf := max.Floats()
		if cos < minCosine {
			err = fmt.Errorf("%s shapes match, but not similar enough:  %v  min_difference=%v max_difference=%v", k, cos, minf, maxf)
			return
		}

		slog.Info("XXX tensors are similar", k, cos, "shape", t.Shape(), "min_difference", minf, "max_difference", maxf)
	}
	err = nil

	return
}

