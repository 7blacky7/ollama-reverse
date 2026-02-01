// Package ggml - GGUF Type-Konstanten
//
// Dieses Modul definiert die GGUF-Datentyp-Konstanten:
// - ggufTypeUint8 bis ggufTypeFloat64: Primitive Datentypen
// - ggufTypeArray: Array-Container-Typ
// - ggufTypeString: String-Typ
package ggml

// GGUF Type Constants - Identifikatoren fuer die verschiedenen Datentypen
const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
)
