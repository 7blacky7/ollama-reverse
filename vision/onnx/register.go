//go:build vision && cgo

// MODUL: onnx/register
// ZWECK: Registriert ONNX Encoder in der globalen Vision Registry
// INPUT: Keine
// OUTPUT: Keine
// NEBENEFFEKTE: Registriert "onnx" Factory bei Package-Import
// ABHAENGIGKEITEN: vision (DefaultRegistry)
// HINWEISE: Import mit _ "github.com/ollama/ollama/vision/onnx"

package onnx

import (
	"github.com/ollama/ollama/vision"
)

func init() {
	vision.DefaultRegistry.Register("onnx", OnnxEncoderFactory)
}
