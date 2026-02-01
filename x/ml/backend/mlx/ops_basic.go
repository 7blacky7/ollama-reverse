//go:build mlx

// Package mlx - Grundlegende Array-Operationen
//
// Hauptfunktionen:
// - Scale, Softmax: Skalierung und Normalisierung
// - SliceUpdate, SliceUpdateDynamic: Slice-Operationen
// - PutAlongAxis, Scatter: Element-Einfügung
// - Copy, Add, Sub: Basis-Arithmetik
// - Max, Min: Aggregationen

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"unsafe"

	"github.com/ollama/ollama/x/ml"
)

// Scale multipliziert das Array mit einem Skalar
func (a *Array) Scale(ctx ml.Context, s float64) ml.Tensor {
	scale := C.mlx_array_new_float(C.float(s))
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		scale,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Softmax berechnet die Softmax-Funktion
func (a *Array) Softmax(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	C.mlx_softmax(
		&r,
		a.a,
		false, // TODO - precise?
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// SliceUpdate aktualisiert einen Slice des Arrays
func (a *Array) SliceUpdate(ctx ml.Context, update ml.Tensor, start, stop, strides []int) ml.Tensor {
	cStart := make([]C.int, len(start))
	for i := range start {
		cStart[i] = C.int(start[i])
	}
	cStop := make([]C.int, len(stop))
	for i := range stop {
		cStop[i] = C.int(stop[i])
	}
	cStrides := make([]C.int, len(strides))
	for i := range strides {
		cStrides[i] = C.int(strides[i])
	}
	var r C.mlx_array
	C.mlx_slice_update(
		&r,
		a.a,
		update.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")
}

// SliceUpdateDynamic aktualisiert einen dynamischen Slice
func (a *Array) SliceUpdateDynamic(ctx ml.Context, update, start ml.Tensor, axes []int) ml.Tensor {
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}

	var r C.mlx_array
	C.mlx_slice_update_dynamic(
		&r,
		a.a,
		update.(*Array).a,
		start.(*Array).a,
		(*C.int)(unsafe.Pointer(&cAxes[0])),
		C.size_t(len(cAxes)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")

}

// PutAlongAxis fügt Werte entlang einer Achse ein
func (a *Array) PutAlongAxis(ctx ml.Context, indicies, values ml.Tensor, axis int) ml.Tensor {
	var r C.mlx_array
	C.mlx_put_along_axis(
		&r,
		a.a,
		indicies.(*Array).a,
		values.(*Array).a,
		C.int(axis),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")
}

// Scatter streut Updates an indizierte Positionen
func (a *Array) Scatter(ctx ml.Context, indicies []ml.Tensor, updates ml.Tensor, axes []int) ml.Tensor {

	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
	}
	indiciesVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(indiciesVec)
	for _, ind := range indicies {
		C.mlx_vector_array_append_value(indiciesVec, ind.(*Array).a)
	}

	var r C.mlx_array
	C.mlx_scatter(
		&r,
		a.a,
		indiciesVec,
		updates.(*Array).a,
		cAxes0,
		C.size_t(len(cAxes)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays[i] = r
			return a
		}
	}
	panic("unable to locate array in context")

}

// Copy kopiert ein Array in ein anderes
func (a *Array) Copy(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	C.mlx_copy(
		&a2.(*Array).a,
		a.a,
		ctx.(*Context).stream,
	)
	// TODO - view?
	return newArray(ctx.(*Context), a2.(*Array).a)
}

// Add addiert zwei Arrays
func (a *Array) Add(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_add(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Sub subtrahiert ein Array von einem anderen
func (a *Array) Sub(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_subtract(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Max berechnet das Maximum entlang der Achsen
func (a *Array) Max(ctx ml.Context, axes []int, keepDims bool) ml.Tensor {
	var r C.mlx_array
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
		C.mlx_max_axes(
			&r,
			a.a,
			cAxes0,
			C.size_t(len(cAxes)),
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	} else {
		C.mlx_max(
			&r,
			a.a,
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)

	}

	return newArray(ctx.(*Context), r)
}

// Min berechnet das Minimum entlang der Achsen
func (a *Array) Min(ctx ml.Context, axes []int, keepDims bool) ml.Tensor {
	var r C.mlx_array
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
		C.mlx_min_axes(
			&r,
			a.a,
			cAxes0,
			C.size_t(len(cAxes)),
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	} else {
		C.mlx_min(
			&r,
			a.a,
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	}

	return newArray(ctx.(*Context), r)
}
