//go:build vision && !cgo

// MODUL: onnx/stub
// ZWECK: Stub-Implementierung wenn CGO nicht verfuegbar ist
// HINWEISE: Gibt Fehler zurueck bei allen Operationen

package onnx

import (
	"errors"

	"github.com/ollama/ollama/vision"
)

// ErrCGORequired wird zurueckgegeben wenn CGO nicht verfuegbar ist
var ErrCGORequired = errors.New("onnx: CGO required but not available")

// OnnxEncoder Stub
type OnnxEncoder struct{}

// OnnxOptions Stub
type OnnxOptions struct {
	ImageSize    int
	EmbeddingDim int
	InputName    string
	OutputName   string
	UseGPU       bool
	NumThreads   int
}

// DefaultOnnxOptions Stub
func DefaultOnnxOptions() OnnxOptions {
	return OnnxOptions{}
}

// NewOnnxEncoder Stub - gibt immer Fehler zurueck
func NewOnnxEncoder(modelPath string, loadOpts vision.LoadOptions) (*OnnxEncoder, error) {
	return nil, ErrCGORequired
}

// Encode Stub
func (e *OnnxEncoder) Encode(imageData []byte) ([]float32, error) {
	return nil, ErrCGORequired
}

// EncodeBatch Stub
func (e *OnnxEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
	return nil, ErrCGORequired
}

// Close Stub
func (e *OnnxEncoder) Close() error {
	return nil
}

// ModelInfo Stub
func (e *OnnxEncoder) ModelInfo() vision.ModelInfo {
	return vision.ModelInfo{}
}

// OnnxEncoderFactory Stub
func OnnxEncoderFactory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return nil, ErrCGORequired
}
