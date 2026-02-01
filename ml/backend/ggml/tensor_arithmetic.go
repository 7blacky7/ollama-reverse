// tensor_arithmetic.go - Basis-Arithmetik-Operationen für Tensoren
// Enthält: Add, Sub, Mul, Div, Scale, Repeat, Stack, Concat, SumRows

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"unsafe"

	"github.com/ollama/ollama/ml"
)

// Add addiert zwei Tensoren elementweise
func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_add(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// Sub subtrahiert zwei Tensoren elementweise
func (t *Tensor) Sub(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sub(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// Mul multipliziert zwei Tensoren elementweise
func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// Div dividiert zwei Tensoren elementweise
func (t *Tensor) Div(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_div(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// Scale skaliert den Tensor mit einem Skalarwert
func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_scale(ctx.(*Context).ctx, t.t, (C.float)(s)),
	}
}

// Repeat wiederholt den Tensor entlang einer Dimension
func (t *Tensor) Repeat(ctx ml.Context, dim, n int) ml.Tensor {
	if dim < 0 || dim >= C.GGML_MAX_DIMS {
		panic("invalid dimension")
	}

	shape := make([]C.int64_t, C.GGML_MAX_DIMS)
	for i := range C.GGML_MAX_DIMS {
		if i == dim {
			shape[i] = C.int64_t(t.Dim(i) * n)
		} else {
			shape[i] = C.int64_t(t.Dim(i))
		}
	}

	tmpl := C.ggml_new_tensor(ctx.(*Context).ctx, t.t._type, C.int(len(shape)), unsafe.SliceData(shape))
	return &Tensor{
		b: t.b,
		t: C.ggml_repeat(ctx.(*Context).ctx, t.t, tmpl),
	}
}

// Stack stapelt Tensoren entlang einer Dimension
func (t *Tensor) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	if len(s) > 0 {
		return t.Concat(ctx, s[0].Stack(ctx, dim, s[1:]...), dim)
	}

	return t
}

// Concat verbindet zwei Tensoren entlang einer Dimension
func (t *Tensor) Concat(ctx ml.Context, t2 ml.Tensor, dim int) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_concat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, C.int(dim)),
	}
}

// SumRows summiert alle Zeilen eines Tensors
func (t *Tensor) SumRows(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_sum_rows(ctx.(*Context).ctx, t.t),
	}
}

// AddID addiert mit Index-Tensor (für Mixture-of-Experts)
func (t *Tensor) AddID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_add_id(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, ids.(*Tensor).t),
	}
}
