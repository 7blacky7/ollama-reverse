// tensor_util.go - Tensor Hilfsfunktionen und Statistik
// Enthält: TopK, Argsort, Mean, Variance, Slice, Chunk, Interpolate

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"github.com/ollama/ollama/ml"
)

// TopK gibt die Top-K Indizes zurück
func (t *Tensor) TopK(ctx ml.Context, k int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_argsort_top_k(ctx.(*Context).ctx, t.t, C.int(k)),
	}
}

// Argsort gibt sortierte Indizes zurück
func (t *Tensor) Argsort(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_argsort(ctx.(*Context).ctx, t.t, C.GGML_SORT_ORDER_ASC),
	}
}

// Mean berechnet den Mittelwert
func (t *Tensor) Mean(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mean(ctx.(*Context).ctx, t.t),
	}
}

// Variance berechnet die Varianz
func (t *Tensor) Variance(ctx ml.Context) ml.Tensor {
	return t.Add(ctx, t.Mean(ctx).Scale(ctx, -1)).
		Sqr(ctx).
		SumRows(ctx).
		Scale(ctx, 1/float64(t.Dim(0)))
}

// Stddev berechnet die Standardabweichung
func (t *Tensor) Stddev(ctx ml.Context) ml.Tensor {
	return t.Variance(ctx).Sqrt(ctx)
}

// Sqr berechnet das Quadrat elementweise
func (t *Tensor) Sqr(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sqr(ctx.(*Context).ctx, t.t),
	}
}

// Sqrt berechnet die Wurzel elementweise
func (t *Tensor) Sqrt(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sqrt(ctx.(*Context).ctx, t.t),
	}
}

// Interpolate führt Interpolation durch
func (t *Tensor) Interpolate(ctx ml.Context, dims [4]int, samplingMode ml.SamplingMode) ml.Tensor {
	var mode C.uint32_t
	switch samplingMode {
	case ml.SamplingModeNearest:
		mode = C.GGML_SCALE_MODE_NEAREST
	case ml.SamplingModeBilinear:
		mode = C.GGML_SCALE_MODE_BILINEAR
	default:
		panic("unsupported interpolate mode")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_interpolate(ctx.(*Context).ctx, t.t, C.int64_t(dims[0]), C.int64_t(dims[1]), C.int64_t(dims[2]), C.int64_t(dims[3]), mode),
	}
}

// Slice gibt eine Ansicht des Tensors entlang einer Dimension zurück
func (t *Tensor) Slice(ctx ml.Context, dim int, low, high, step int) ml.Tensor {
	if dim < 0 || dim >= C.GGML_MAX_DIMS {
		panic("invalid dimension")
	} else if low < 0 || high > t.Dim(dim) || low >= high || step < 1 {
		panic("invalid slice parameters")
	}

	if dim == 0 && step > 1 {
		// Spezialfall für dim=0, step>1
		return t.View(ctx,
			low*t.Stride(0), 1,
			step*t.Stride(0), (high-low+1)/step,
			t.Stride(1), t.Dim(1),
			// Erhalt von dim 3 durch Zusammenführen mit dim 2
			t.Stride(2), t.Dim(2)*t.Dim(3),
		).Contiguous(ctx, (high-low+1)/step, t.Dim(1), t.Dim(2), t.Dim(3))
	}

	args := []int{
		low * t.Stride(dim), t.Dim(0),
		t.Stride(1), t.Dim(1),
		t.Stride(2), t.Dim(2),
		t.Stride(3), t.Dim(3),
	}

	if step == 1 {
		args[dim*2+1] = high - low
		return t.View(ctx, args[0], args[1:]...)
	}

	args[dim*2] = step * t.Stride(dim)
	args[dim*2+1] = (high - low + 1) / step
	return t.View(ctx, args[0], args[1:]...)
}

// Chunk teilt den Tensor in gleichgroße Teile
func (t *Tensor) Chunk(ctx ml.Context, dim, chunk int) []ml.Tensor {
	sections := make([]int, 0, t.Dim(dim)/chunk+1)
	for rest := t.Dim(dim); rest > 0; rest -= chunk {
		sections = append(sections, min(chunk, rest))
	}
	return t.ChunkSections(ctx, dim, sections...)
}

// ChunkSections teilt den Tensor in Abschnitte
func (t *Tensor) ChunkSections(ctx ml.Context, dim int, sections ...int) []ml.Tensor {
	var offset int
	s := make([]ml.Tensor, len(sections))
	for i, section := range sections {
		s[i] = t.Slice(ctx, dim, offset, offset+section, 1)
		offset += section
	}
	if offset != t.Dim(dim) {
		panic("sections do not sum to tensor dimension")
	}
	return s
}
