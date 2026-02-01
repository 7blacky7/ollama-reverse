// context.go - Context und Tensor Interfaces fuer ML-Operationen
// Dieses Modul definiert die Schnittstellen fuer Tensor-Operationen und Compute-Kontexte.
package ml

// Context represents an execution context for tensor operations.
type Context interface {
	Empty(dtype DType, shape ...int) Tensor
	Zeros(dtype DType, shape ...int) Tensor
	FromBytes(dtype DType, s []byte, shape ...int) Tensor
	FromFloats(s []float32, shape ...int) Tensor
	FromInts(s []int32, shape ...int) Tensor

	// Arange creates a 1D tensor with values within an interval (start, stop] increased by step.
	Arange(start, stop, step float32, dtype DType) Tensor

	Forward(...Tensor) Context

	// SetBatchSize provides a hint on the batch size to optimize processing
	// Uses heuristics if not set
	SetBatchSize(int)

	Compute(...Tensor)
	ComputeWithNotify(func(), ...Tensor) // notify callback once compute has begun

	// Reserve is analogous to Compute but rather than executing a
	// graph, simply preallocates memory. Typically called with a
	// worst case graph to ensure all resources are available for
	// for future inference.
	Reserve()

	MaxGraphNodes() int
	Close()

	// Input returns a context appropriate for creating tensors that are
	// inputs to the model (which includes things like output locations)
	Input() Context

	// Layer returns a context appropriate for creating intermediate tensors
	Layer(int) Context
}

// Tensor represents a multi-dimensional array with various operations.
type Tensor interface {
	Dim(n int) int
	Stride(n int) int

	Shape() []int
	DType() DType
	Cast(ctx Context, dtype DType) Tensor

	Bytes() []byte
	Floats() []float32

	FromBytes([]byte)
	FromFloats([]float32)
	FromInts([]int32)

	Add(ctx Context, t2 Tensor) Tensor
	Sub(ctx Context, t2 Tensor) Tensor
	Mul(ctx Context, t2 Tensor) Tensor
	Div(ctx Context, t2 Tensor) Tensor

	Mulmat(ctx Context, t2 Tensor) Tensor
	MulmatFullPrec(ctx Context, t2 Tensor) Tensor
	MulmatID(ctx Context, t2, ids Tensor) Tensor
	AddID(ctx Context, t2, ids Tensor) Tensor

	Softmax(ctx Context) Tensor
	L2Norm(ctx Context, eps float32) Tensor
	LayerNorm(ctx Context, weight, bias Tensor, eps float32) Tensor
	RMSNorm(ctx Context, weight Tensor, eps float32) Tensor
	Scale(ctx Context, s float64) Tensor
	SumRows(ctx Context) Tensor

	AvgPool2D(ctx Context, k, s int, p float32) Tensor
	Conv2D(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor
	Conv3D(ctx Context, weight Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) Tensor
	SSMConv(ctx Context, kernel Tensor) Tensor

	IM2Col(ctx Context, weight Tensor, s0, s1, p0, p1, d0, d1 int) Tensor

	Sin(ctx Context) Tensor
	Cos(ctx Context) Tensor
	Tanh(ctx Context) Tensor
	GELU(ctx Context, up ...Tensor) Tensor
	QuickGELU(ctx Context, up ...Tensor) Tensor
	SILU(ctx Context, up ...Tensor) Tensor
	RELU(ctx Context, up ...Tensor) Tensor
	Sigmoid(ctx Context) Tensor

	// AlphaLimitSILU is a variant of SILU that clamps the input to the range [-limit, limit]
	SILUAlphaLimit(ctx Context, up Tensor, alpha, limit float32) Tensor

	Reshape(ctx Context, shape ...int) Tensor
	View(ctx Context, offset int, shape ...int) Tensor
	Permute(ctx Context, shape ...int) Tensor
	Contiguous(ctx Context, shape ...int) Tensor

	Pad(ctx Context, shape ...int) Tensor

	Stack(ctx Context, dim int, s ...Tensor) Tensor

	// Repeat repeats the tensor n times along dimension dim
	Repeat(ctx Context, dim, n int) Tensor
	Concat(ctx Context, t2 Tensor, dim int) Tensor
	Rows(ctx Context, t2 Tensor) Tensor
	SetRows(ctx Context, src Tensor, idxs Tensor) Tensor
	Copy(ctx Context, t2 Tensor) Tensor
	Duplicate(ctx Context) Tensor

	Slice(ctx Context, dim, low, high, step int) Tensor
	Chunk(ctx Context, dim int, size int) []Tensor
	ChunkSections(ctx Context, dim int, sections ...int) []Tensor

	TopK(ctx Context, k int) Tensor
	Argsort(ctx Context) Tensor
	Mean(ctx Context) Tensor
	Variance(ctx Context) Tensor
	Stddev(ctx Context) Tensor
	Sqr(ctx Context) Tensor
	Sqrt(ctx Context) Tensor

	Interpolate(ctx Context, dims [4]int, samplingMode SamplingMode) Tensor
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
//
// cacheConfigApplied indicates whether the optimizations requested through CacheConfig have been performed
type ScaledDotProductAttention interface {
	ScaledDotProductAttention(ctx Context, key, value, mask, sinks Tensor, vmla Tensor, scale float64, cacheConfigApplied bool) Tensor
}
