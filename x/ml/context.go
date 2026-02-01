// Package ml - Context Interface
// Dieses Modul definiert das Context-Interface für Tensor-Operationen
// und Graph-Ausführung im ML-Backend.
package ml

type Context interface {
	Empty(dtype DType, shape ...int) Tensor
	Zeros(dtype DType, shape ...int) Tensor
	// FromBytes(dtype DType, s []byte, shape ...int) Tensor
	FromFloats(s []float32, shape ...int) Tensor
	FromInts(s []int32, shape ...int) Tensor
	RandomNormal(shape []int, dtype DType, loc, scale float32, key Tensor) Tensor

	// Arange creates a 1D tensor with values within an interval (start, stop] increased by step.
	Arange(start, stop, step float32, dtype DType) Tensor

	Forward(...Tensor) Context

	// SetBatchSize provides a hint on the batch size to optimize processing
	// Uses heuristics if not set
	// SetBatchSize(int)

	Compute(...Tensor)
	// ComputeWithNotify(func(), ...Tensor) // notify callback once compute has begun

	// Reserve is analogous to Compute but rather than executing a
	// graph, simply preallocates memory. Typically called with a
	// worst case graph to ensure all resources are available for
	// for future inference.
	// Reserve()

	// MaxGraphNodes() int
	Close()

	// Input returns a context appropriate for creating tensors that are
	// inputs to the model (which includes things like output locations)
	Input() Context

	// Layer returns a context appropriate for creating intermediate tensors
	Layer(int) Context

	// Load a tensor from "filename" safetensors file, and compare with the input tensor
	// Returns error if the shape is inconsistent, or similarity measures are below 99%
	CompareWith(filename string, tensors map[string]Tensor, abortOnError bool) error
}
