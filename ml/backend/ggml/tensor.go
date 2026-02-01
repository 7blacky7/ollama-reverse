// tensor.go - Tensor-Struktur und Basis-Methoden
// Enthält: Tensor struct, Shape, Bytes, Floats, DType, Cast

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"encoding/binary"
	"log/slog"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

// Tensor repräsentiert einen GGML-Tensor
type Tensor struct {
	b    *Backend
	t    *C.struct_ggml_tensor
	sync func()
}

// LogValue gibt den Tensor als slog-Wert zurück
func (t *Tensor) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", C.GoString(C.ggml_get_name(t.t))),
		slog.String("type", C.GoString(C.ggml_type_name(t.t._type))),
		slog.Any("shape", t.Shape()),
	)
}

// Dim gibt die Größe einer Dimension zurück
func (t *Tensor) Dim(n int) int {
	return int(t.t.ne[n])
}

// Stride gibt den Stride einer Dimension zurück
func (t *Tensor) Stride(n int) int {
	return int(t.t.nb[n])
}

// Shape gibt die Form des Tensors zurück
func (t *Tensor) Shape() []int {
	shape := make([]int, C.ggml_n_dims(t.t))
	for i := range shape {
		shape[i] = t.Dim(i)
	}

	return shape
}

// Bytes gibt die Tensor-Daten als Bytes zurück
func (t *Tensor) Bytes() (data []byte) {
	if t.sync != nil {
		data = make([]byte, C.ggml_nbytes(t.t))

		t.sync()
		C.ggml_backend_tensor_get(t.t, unsafe.Pointer(&data[0]), 0, C.ggml_nbytes(t.t))
	}

	return
}

// Floats gibt die Tensor-Daten als Float32 zurück
func (t *Tensor) Floats() (data []float32) {
	if t.sync != nil {
		data = make([]float32, C.ggml_nelements(t.t))

		t.sync()
		C.ggml_backend_tensor_get(t.t, unsafe.Pointer(&data[0]), 0, C.ggml_nbytes(t.t))
	}

	return
}

// tensorSet setzt Tensor-Daten aus einem Slice
func tensorSet[S ~[]E, E byte | float32 | int32](t *Tensor, s S) {
	if len(s) == 0 {
		return
	}
	if int(C.ggml_nbytes(t.t)) != len(s)*binary.Size(s[0]) {
		panic("data size does not match tensor size")
	}
	C.ggml_backend_tensor_set(t.t, unsafe.Pointer(&s[0]), 0, C.ggml_nbytes(t.t))
}

// FromBytes setzt Tensor-Daten aus Bytes
func (t *Tensor) FromBytes(s []byte) {
	tensorSet(t, s)
}

// FromFloats setzt Tensor-Daten aus Float32
func (t *Tensor) FromFloats(s []float32) {
	tensorSet(t, s)
}

// FromInts setzt Tensor-Daten aus Int32
func (t *Tensor) FromInts(s []int32) {
	tensorSet(t, s)
}

// DType gibt den Datentyp des Tensors zurück
func (t *Tensor) DType() ml.DType {
	switch t.t._type {
	case C.GGML_TYPE_F32:
		return ml.DTypeF32
	case C.GGML_TYPE_F16:
		return ml.DTypeF16
	case C.GGML_TYPE_Q8_0:
		return ml.DTypeQ80
	case C.GGML_TYPE_Q4_0:
		return ml.DTypeQ40
	case C.GGML_TYPE_I32:
		return ml.DTypeI32
	case C.GGML_TYPE_MXFP4:
		return ml.DTypeMXFP4
	default:
		return ml.DTypeOther
	}
}

// Cast konvertiert den Tensor zu einem anderen Datentyp
func (t *Tensor) Cast(ctx ml.Context, dtype ml.DType) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cast(ctx.(*Context).ctx, t.t, ggmlDType(dtype)),
	}
}

// Copy kopiert Daten in einen anderen Tensor
func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
	}
}

// Duplicate dupliziert den Tensor
func (t *Tensor) Duplicate(ctx ml.Context) ml.Tensor {
	return &Tensor{
		b: t.b,
		t: C.ggml_dup(ctx.(*Context).ctx, t.t),
	}
}
