// Package ml - Datentypen
// Dieses Modul definiert die grundlegenden Datentypen (DType)
// und Sampling-Modi fuer das ML-Backend.
package ml

type DType int

const (
	DTypeBool DType = iota
	DTypeUint8
	DTypeUint16
	DTypeUint32
	DTypeUint64
	DTypeInt8
	DTypeInt16
	DTypeInt32
	DTypeInt64
	DTypeFloat16
	DTypeFloat32
	DTypeFloat64
	DTypeBfloat16
	DTypeComplex64
)

type SamplingMode int

const (
	SamplingModeNearest SamplingMode = iota
	SamplingModeBilinear
)
