// tensor_matrix.go - Matrix-Operationen und Normalisierungen
// Enthält: Mulmat, MulmatFullPrec, MulmatID, L2Norm, LayerNorm, RMSNorm

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"github.com/ollama/ollama/ml"
)

// Mulmat führt Matrix-Multiplikation durch
// Bei Shape [m, p, ...] und t2 [m, n, ...] ergibt sich [p, n, ...]
func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// MulmatFullPrec führt Matrix-Multiplikation mit voller Präzision durch
// Verwendet F32 statt der Standard-Präzision
func (t *Tensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	mul := C.ggml_mul_mat(ctx.(*Context).ctx, t.t, t2.(*Tensor).t)
	C.ggml_mul_mat_set_prec(mul, C.GGML_PREC_F32)

	return &Tensor{
		b: t.b,
		t: mul,
	}
}

// MulmatID führt Matrix-Multiplikation mit Index-Tensor durch
// Wird für Mixture-of-Experts (MoE) Modelle verwendet
func (t *Tensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_mul_mat_id(ctx.(*Context).ctx, t.t, t2.(*Tensor).t, ids.(*Tensor).t),
	}
}

// L2Norm berechnet die L2-Normalisierung des Tensors
// eps ist der Epsilon-Wert zur Vermeidung von Division durch Null
func (t *Tensor) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_l2_norm(ctx.(*Context).ctx, t.t, C.float(eps)),
	}
}

// LayerNorm führt Layer-Normalisierung durch
// w ist der Weight-Tensor, b ist der Bias-Tensor (beide optional)
// eps ist der Epsilon-Wert für numerische Stabilität
func (t *Tensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	tt := C.ggml_norm(ctx.(*Context).ctx, t.t, C.float(eps))
	if w != nil {
		tt = C.ggml_mul(ctx.(*Context).ctx, tt, w.(*Tensor).t)
		if b != nil {
			tt = C.ggml_add(ctx.(*Context).ctx, tt, b.(*Tensor).t)
		}
	}

	return &Tensor{b: t.b, t: tt}
}

// RMSNorm führt Root Mean Square Normalisierung durch
// w ist der Weight-Tensor (optional)
// eps ist der Epsilon-Wert für numerische Stabilität
func (t *Tensor) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	tt := C.ggml_rms_norm(ctx.(*Context).ctx, t.t, C.float(eps))
	if w != nil {
		tt = C.ggml_mul(ctx.(*Context).ctx, tt, w.(*Tensor).t)
	}

	return &Tensor{b: t.b, t: tt}
}
