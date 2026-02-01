// dump.go - Dump-Funktionen fuer Tensor-Debugging und Visualisierung
// Dieses Modul stellt Hilfsfunktionen zum Ausgeben von Tensor-Inhalten bereit.
package ml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

type number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

func mul[T number](s ...T) T {
	p := T(1)
	for _, v := range s {
		p *= v
	}

	return p
}

// DumpOptions configures tensor dump output format.
type DumpOptions func(*dumpOptions)

// DumpWithPrecision sets the number of decimal places to print. Applies to float32 and float64.
func DumpWithPrecision(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.Precision = n
	}
}

// DumpWithThreshold sets the threshold for printing the entire tensor. If the number of elements
// is less than or equal to this value, the entire tensor will be printed. Otherwise, only the
// beginning and end of each dimension will be printed.
func DumpWithThreshold(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.Threshold = n
	}
}

// DumpWithEdgeItems sets the number of elements to print at the beginning and end of each dimension.
func DumpWithEdgeItems(n int) DumpOptions {
	return func(opts *dumpOptions) {
		opts.EdgeItems = n
	}
}

type dumpOptions struct {
	Precision, Threshold, EdgeItems int
}

// Dump converts a tensor to a human-readable string representation.
func Dump(ctx Context, t Tensor, optsFuncs ...DumpOptions) string {
	opts := dumpOptions{Precision: 4, Threshold: 1000, EdgeItems: 3}
	for _, optsFunc := range optsFuncs {
		optsFunc(&opts)
	}

	if mul(t.Shape()...) <= opts.Threshold {
		opts.EdgeItems = math.MaxInt
	}

	switch t.DType() {
	case DTypeF32:
		return dump[[]float32](ctx, t, opts.EdgeItems, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
		})
	case DTypeF16, DTypeQ80, DTypeQ40:
		f32 := ctx.Input().Empty(DTypeF32, t.Shape()...)
		f32 = t.Copy(ctx, f32)
		return dump[[]float32](ctx, f32, opts.EdgeItems, func(f float32) string {
			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
		})
	case DTypeI32:
		return dump[[]int32](ctx, t, opts.EdgeItems, func(i int32) string {
			return strconv.FormatInt(int64(i), 10)
		})
	default:
		return "<unsupported>"
	}
}

func dump[S ~[]E, E number](ctx Context, t Tensor, items int, fn func(E) string) string {
	if t.Bytes() == nil {
		ctx.Forward(t).Compute(t)
	}

	s := make(S, mul(t.Shape()...))
	if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &s); err != nil {
		panic(err)
	}

	shape := t.Shape()
	slices.Reverse(shape)

	var sb strings.Builder
	var f func([]int, int)
	f = func(dims []int, stride int) {
		prefix := strings.Repeat(" ", len(shape)-len(dims)+1)
		sb.WriteString("[")
		defer func() { sb.WriteString("]") }()
		for i := 0; i < dims[0]; i++ {
			if i >= items && i < dims[0]-items {
				sb.WriteString("..., ")
				// skip to next printable element
				skip := dims[0] - 2*items
				if len(dims) > 1 {
					stride += mul(append(dims[1:], skip)...)
					fmt.Fprint(&sb, strings.Repeat("\n", len(dims)-1), prefix)
				}
				i += skip - 1
			} else if len(dims) > 1 {
				f(dims[1:], stride)
				stride += mul(dims[1:]...)
				if i < dims[0]-1 {
					fmt.Fprint(&sb, ",", strings.Repeat("\n", len(dims)-1), prefix)
				}
			} else {
				text := fn(s[stride+i])
				if len(text) > 0 && text[0] != '-' {
					sb.WriteString(" ")
				}

				sb.WriteString(text)
				if i < dims[0]-1 {
					sb.WriteString(", ")
				}
			}
		}
	}
	f(shape, 0)

	return sb.String()
}
