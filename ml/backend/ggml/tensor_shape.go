// tensor_shape.go - Shape-Operationen für Tensoren
// Enthält: Reshape, View, Permute, Contiguous, Pad, Rows, SetRows

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"slices"

	"github.com/ollama/ollama/ml"
)

// Contiguous erstellt eine zusammenhängende Kopie des Tensors
// Optional kann eine neue Shape angegeben werden
func (t *Tensor) Contiguous(ctx ml.Context, shape ...int) ml.Tensor {
	if slices.Contains(shape, -1) {
		inferShape(t, shape)
	}

	switch len(shape) {
	case 0:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont(ctx.(*Context).ctx, t.t),
		}
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			b: t.b,
			t: C.ggml_cont_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

// Reshape ändert die Form des Tensors ohne Datenkopie
// Verwendet Contiguous falls der Tensor nicht zusammenhängend ist
func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	if !C.ggml_is_contiguous(t.t) {
		return t.Contiguous(ctx, shape...)
	}

	if slices.Contains(shape, -1) {
		inferShape(t, shape)
	}

	switch len(shape) {
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0])),
		}
	case 2:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_2d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1])),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_3d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2])),
		}
	case 4:
		return &Tensor{
			b: t.b,
			t: C.ggml_reshape_4d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.int64_t(shape[1]), C.int64_t(shape[2]), C.int64_t(shape[3])),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

// View erstellt eine Ansicht auf einen Teil des Tensors
// Die Shape-Parameter hängen von der Dimensionalität ab:
// 1D: [ne0]
// 2D: [ne0, nb1, ne1]
// 3D: [ne0, nb1, ne1, nb2, ne2]
// 4D: [ne0, nb1, ne1, nb2, ne2, nb3, ne3]
func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	switch len(shape) {
	case 1:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_1d(ctx.(*Context).ctx, t.t, C.int64_t(shape[0]), C.size_t(offset)),
		}
	case 3:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_2d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]),
				C.size_t(shape[1]),
				C.size_t(offset)),
		}
	case 5:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_3d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]),
				C.size_t(shape[1]), C.size_t(shape[3]),
				C.size_t(offset)),
		}
	case 7:
		return &Tensor{
			b: t.b,
			t: C.ggml_view_4d(ctx.(*Context).ctx, t.t,
				C.int64_t(shape[0]), C.int64_t(shape[2]), C.int64_t(shape[4]), C.int64_t(shape[6]),
				C.size_t(shape[1]), C.size_t(shape[3]), C.size_t(shape[5]),
				C.size_t(offset)),
		}
	default:
		panic("unsupported number of dimensions")
	}
}

// Permute permutiert die Dimensionen des Tensors
// Erwartet die neue Reihenfolge der Dimensionen
func (t *Tensor) Permute(ctx ml.Context, order ...int) ml.Tensor {
	if len(order) != len(t.Shape()) && len(order) != 4 {
		panic("invalid number of dimensions for permute")
	}

	// ggml_permute erfordert immer 4 Dimensionen
	for i := len(order); i < 4; i++ {
		order = append(order, i)
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_permute(ctx.(*Context).ctx, t.t, C.int(order[0]), C.int(order[1]), C.int(order[2]), C.int(order[3])),
	}
}

// Pad fügt Padding zu den Dimensionen hinzu
// Erwartet genau 4 Padding-Werte, wobei shape[3] 0 sein muss (CUDA-Limitierung)
func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) != 4 {
		panic("expected 4 dimensions")
	} else if shape[3] != 0 {
		panic("cuda does not support 4d tensors")
	}

	return &Tensor{
		b: t.b,
		t: C.ggml_pad(ctx.(*Context).ctx, t.t, C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])),
	}
}

// Rows gibt Zeilen nach Indizes zurück (Embedding-Lookup)
func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_get_rows(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// SetRows setzt Zeilen nach Indizes
func (t *Tensor) SetRows(ctx ml.Context, src ml.Tensor, idxs ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_set_rows(ctx.(*Context).ctx, t.t, src.(*Tensor).t, idxs.(*Tensor).t),
	}
}
