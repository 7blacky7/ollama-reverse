// Package ml - Tensor Interface
// Dieses Modul definiert das Tensor-Interface für alle mathematischen
// Operationen sowie RoPE-Optionen für Positionskodierung.
package ml

type RoPEOptions struct {
	Base  *float32
	Freqs Tensor
}

func WithRoPEBase(base float32) func(*RoPEOptions) {
	return func(opts *RoPEOptions) {
		opts.Base = &base
	}
}

func WithRoPEFreqs(freqs Tensor) func(*RoPEOptions) {
	return func(opts *RoPEOptions) {
		opts.Freqs = freqs
	}
}

type Tensor interface {
	ToString() string
	RoPE(ctx Context, dims int, traditional bool, scale float32, offset int, options ...func(*RoPEOptions)) Tensor
	ScaledDotProductAttention(ctx Context, keys, values Tensor, scale float64, maskMode string, mask Tensor, sinks Tensor) Tensor
	TakeAxes(ctx Context, indicies Tensor, axes int) Tensor
	// TakeAxes(ctx Context, axes int, indicies ...int) Tensor

	Dim(n int) int
	Stride(n int) int

	Shape() []int
	DType() DType
	// Cast(ctx Context, dtype DType) Tensor

	// Bytes() []byte
	Floats() []float32
	Ints() []int32

	// FromBytes([]byte)
	// FromFloats([]float32)
	// FromInts([]int32)

	Add(ctx Context, t2 Tensor) Tensor
	Sub(ctx Context, t2 Tensor) Tensor
	// Mul(ctx Context, t2 Tensor) Tensor
	// Div(ctx Context, t2 Tensor) Tensor

	Max(ctx Context, axes []int, keepDims bool) Tensor
	Min(ctx Context, axes []int, keepDims bool) Tensor

	Matmul(ctx Context, a2 Tensor) Tensor
	// Mulmat(ctx Context, t2 Tensor) Tensor
	// MulmatFullPrec(ctx Context, t2 Tensor) Tensor
	// MulmatID(ctx Context, t2, ids Tensor) Tensor
	// AddID(ctx Context, t2, ids Tensor) Tensor

	Softmax(ctx Context) Tensor
	L2Norm(ctx Context, eps float32) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor
	// SumRows(ctx Context) Tensor

	AvgPool2D(ctx Context, k, s int, p float32) Tensor
	Conv2D(ctx Context, weight Tensor, stride0, stride1, padding0, padding1, dilation0, dilation1, groups int) Tensor
	Conv3D(ctx Context, weight Tensor, stride0, stride1, stride2, padding0, padding1, padding2, dilation0, dilation1, dilation2, groups int) Tensor

	// IM2Col(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	// Sin(ctx Context) Tensor
	// Cos(ctx Context) Tensor
	// Tanh(ctx Context) Tensor
	GELU(ctx Context, up ...Tensor) Tensor
	// QuickGELU(ctx Context, up ...Tensor) Tensor
	// SILU(ctx Context, up ...Tensor) Tensor
	// RELU(ctx Context, up ...Tensor) Tensor
	// Sigmoid(ctx Context) Tensor

	// AlphaLimitSILU is a variant of SILU that clamps the input to the range [-limit, limit]
	// SILUAlphaLimit(ctx Context, up Tensor, alpha, limit float32) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	AsStrided(ctx Context, shape, strides []int, offset int) Tensor
	Transpose(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context, allowColMajor bool) Tensor

	// Pad(ctx Context, shape ...int) Tensor

	// Stack(ctx Context, dim int, s ...Tensor) Tensor

	// Repeat repeats the tensor n times along dimension dim
	// Repeat(ctx Context, dim, n int) Tensor
	// Concat(ctx Context, t2 Tensor, dim int) Tensor
	// Rows(ctx Context, t2 Tensor) Tensor

	// TODO these probably aren't actually needed - false starts on trying to wire up cache
	// SliceUpdate(ctx Context, update Tensor, start, stop, strides []int) Tensor
	// SliceUpdateDynamic(ctx Context, update, start Tensor, axes []int) Tensor
	// PutAlongAxis(ctx Context, indicies, values Tensor, axis int) Tensor

	Scatter(ctx Context, indicies []Tensor, updates Tensor, axes []int) Tensor

	Copy(ctx Context, t2 Tensor) Tensor
	// Duplicate(ctx Context) Tensor

	// Slice(ctx Context, dim, low, high, step int) Tensor
	// Chunk(ctx Context, dim int, size int) []Tensor
	// ChunkSections(ctx Context, dim int, sections ...int) []Tensor

	// TopK(ctx Context, k int) Tensor
	// Argsort(ctx Context) Tensor
	// Mean(ctx Context) Tensor
	// Variance(ctx Context) Tensor
	// Stddev(ctx Context) Tensor
	// Sqr(ctx Context) Tensor
	// Sqrt(ctx Context) Tensor

	// Interpolate(ctx Context, dims [4]int, samplingMode SamplingMode) Tensor
}

// ScaledDotProductAttention implements a fused attention
// operation equivalent to following code on a tensor named
// query:
//
// query = query.Permute(ctx, 0, 2, 1, 3)
// key = key.Permute(ctx, 0, 2, 1, 3)
// value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
//
// kq := key.MulmatFullPrec(ctx, query)
//
// kq = kq.Scale(ctx, scale)
//
//	if mask != nil {
//		kq = kq.Add(ctx, mask)
//	}
//
// kq = kq.Softmax(ctx)
//
// kqv := value.Mulmat(ctx, kq)
// return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
// type ScaledDotProductAttention interface {
// 	ScaledDotProductAttention(ctx Context, key, value, mask, sinks Tensor, vmla Tensor, scale float64) Tensor
// }

// type number interface {
// 	~int | ~int8 | ~int16 | ~int32 | ~int64 |
// 		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
// 		~float32 | ~float64 |
// 		~complex64 | ~complex128
// }

// func mul[T number](s ...T) T {
// 	p := T(1)
// 	for _, v := range s {
// 		p *= v
// 	}

// 	return p
// }

// type DumpOptions func(*dumpOptions)

// // DumpWithPrecision sets the number of decimal places to print. Applies to float32 and float64.
// func DumpWithPrecision(n int) DumpOptions {
// 	return func(opts *dumpOptions) {
// 		opts.Precision = n
// 	}
// }

// // DumpWithThreshold sets the threshold for printing the entire tensor. If the number of elements
// // is less than or equal to this value, the entire tensor will be printed. Otherwise, only the
// // beginning and end of each dimension will be printed.
// func DumpWithThreshold(n int) DumpOptions {
// 	return func(opts *dumpOptions) {
// 		opts.Threshold = n
// 	}
// }

// // DumpWithEdgeItems sets the number of elements to print at the beginning and end of each dimension.
// func DumpWithEdgeItems(n int) DumpOptions {
// 	return func(opts *dumpOptions) {
// 		opts.EdgeItems = n
// 	}
// }

// type dumpOptions struct {
// 	Precision, Threshold, EdgeItems int
// }

// func Dump(ctx Context, t Tensor, optsFuncs ...DumpOptions) string {
// 	opts := dumpOptions{Precision: 4, Threshold: 1000, EdgeItems: 3}
// 	for _, optsFunc := range optsFuncs {
// 		optsFunc(&opts)
// 	}

// 	if mul(t.Shape()...) <= opts.Threshold {
// 		opts.EdgeItems = math.MaxInt
// 	}

// 	switch t.DType() {
// 	case DTypeFloat32:
// 		return dump[[]float32](ctx, t, opts.EdgeItems, func(f float32) string {
// 			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
// 		})
// 	case DTypeFloat16: // TODO other types...
// 		f32 := ctx.Input().Empty(DTypeFloat32, t.Shape()...)
// 		f32 = t.Copy(ctx, f32)
// 		return dump[[]float32](ctx, f32, opts.EdgeItems, func(f float32) string {
// 			return strconv.FormatFloat(float64(f), 'f', opts.Precision, 32)
// 		})
// 	case DTypeInt32:
// 		return dump[[]int32](ctx, t, opts.EdgeItems, func(i int32) string {
// 			return strconv.FormatInt(int64(i), 10)
// 		})
// 	default:
// 		return "<unsupported>"
// 	}
// }

// func dump[S ~[]E, E number](ctx Context, t Tensor, items int, fn func(E) string) string {
// 	if t.Bytes() == nil {
// 		ctx.Compute(t)
// 	}

// 	s := make(S, mul(t.Shape()...))
// 	if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &s); err != nil {
// 		panic(err)
// 	}

// 	shape := t.Shape()
// 	slices.Reverse(shape)

// 	var sb strings.Builder
// 	var f func([]int, int)
// 	f = func(dims []int, stride int) {
// 		prefix := strings.Repeat(" ", len(shape)-len(dims)+1)
// 		sb.WriteString("[")
// 		defer func() { sb.WriteString("]") }()
// 		for i := 0; i < dims[0]; i++ {
// 			if i >= items && i < dims[0]-items {
// 				sb.WriteString("..., ")
// 				// skip to next printable element
// 				skip := dims[0] - 2*items
// 				if len(dims) > 1 {
// 					stride += mul(append(dims[1:], skip)...)
// 					fmt.Fprint(&sb, strings.Repeat("\n", len(dims)-1), prefix)
// 				}
// 				i += skip - 1
// 			} else if len(dims) > 1 {
// 				f(dims[1:], stride)
// 				stride += mul(dims[1:]...)
// 				if i < dims[0]-1 {
// 					fmt.Fprint(&sb, ",", strings.Repeat("\n", len(dims)-1), prefix)
// 				}
// 			} else {
// 				text := fn(s[stride+i])
// 				if len(text) > 0 && text[0] != '-' {
// 					sb.WriteString(" ")
// 				}

// 				sb.WriteString(text)
// 				if i < dims[0]-1 {
// 					sb.WriteString(", ")
// 				}
// 			}
// 		}
// 	}
// 	f(shape, 0)

// 	return sb.String()
// }
