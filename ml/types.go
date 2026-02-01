// types.go - Datentypen und Konstanten fuer ML-Operationen
// Dieses Modul definiert grundlegende Typen wie DType und SamplingMode.
package ml

// DType represents the data type of tensor elements.
type DType int

const (
	DTypeOther DType = iota
	DTypeF32
	DTypeF16
	DTypeQ80
	DTypeQ40
	DTypeI32
	DTypeMXFP4
)

// SamplingMode specifies the interpolation method for tensor resizing.
type SamplingMode int

const (
	SamplingModeNearest SamplingMode = iota
	SamplingModeBilinear
)
