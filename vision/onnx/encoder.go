//go:build vision && cgo

// MODUL: onnx/encoder
// ZWECK: ONNX Vision Encoder mit VisionEncoder Interface
// INPUT: Modell-Pfad (.onnx), Bild-Daten ([]byte), LoadOptions
// OUTPUT: 768-dim Embedding-Vektoren ([]float32)
// NEBENEFFEKTE: Laedt ONNX Runtime Session, alloziert GPU/CPU Speicher
// ABHAENGIGKEITEN: session.go, preprocess.go, vision (VisionEncoder Interface)
// HINWEISE: Optimiert fuer Nomic Embed Vision v1.5, Thread-sicher

package onnx

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Konstanten
// ============================================================================

const (
	// DefaultEmbeddingDim ist die Standard-Embedding-Dimension fuer Nomic v1.5
	DefaultEmbeddingDim = 768

	// DefaultImageSize ist die Fallback-Bildgroesse wenn nicht aus Modell lesbar
	// 224 ist Standard fuer die meisten Vision Transformer (ViT, Nomic, etc.)
	DefaultImageSize = 224

	// DefaultInputName ist der ONNX Input-Tensor Name
	DefaultInputName = "pixel_values"

	// DefaultOutputName ist der ONNX Output-Tensor Name
	DefaultOutputName = "image_embeds"
)

// ============================================================================
// Fehler-Definitionen
// ============================================================================

var (
	ErrModelLoad     = errors.New("onnx: modell laden fehlgeschlagen")
	ErrSessionCreate = errors.New("onnx: session erstellen fehlgeschlagen")
	ErrInference     = errors.New("onnx: inference fehlgeschlagen")
	ErrPreprocess    = errors.New("onnx: preprocessing fehlgeschlagen")
	ErrAlreadyClosed = errors.New("onnx: encoder bereits geschlossen")
	ErrInvalidInput  = errors.New("onnx: ungueltige eingabe")
)

// ============================================================================
// OnnxEncoder - Hauptstruktur
// ============================================================================

// OnnxEncoder implementiert vision.VisionEncoder mit ONNX Runtime.
// Optimiert fuer Nomic Embed Vision v1.5 mit INT8 Quantisierung.
type OnnxEncoder struct {
	session *Session
	info    vision.ModelInfo
	opts    OnnxOptions
	closed  bool
	mu      sync.RWMutex
}

// OnnxOptions konfiguriert den ONNX Encoder
type OnnxOptions struct {
	ImageSize    int    // Ziel-Bildgroesse (Standard: 384)
	EmbeddingDim int    // Embedding-Dimension (Standard: 768)
	InputName    string // ONNX Input-Tensor Name
	OutputName   string // ONNX Output-Tensor Name
	UseGPU       bool   // GPU-Beschleunigung aktivieren
	NumThreads   int    // CPU Threads (0 = auto)
}

// DefaultOnnxOptions gibt Standard-Optionen zurueck
func DefaultOnnxOptions() OnnxOptions {
	return OnnxOptions{
		ImageSize:    DefaultImageSize,
		EmbeddingDim: DefaultEmbeddingDim,
		InputName:    DefaultInputName,
		OutputName:   DefaultOutputName,
		UseGPU:       false,
		NumThreads:   0,
	}
}

// ============================================================================
// Konstruktor
// ============================================================================

// NewOnnxEncoder erstellt einen neuen ONNX-basierten Vision Encoder.
func NewOnnxEncoder(modelPath string, loadOpts vision.LoadOptions) (*OnnxEncoder, error) {
	// Datei existiert?
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: %s", ErrModelLoad, modelPath)
	}

	opts := DefaultOnnxOptions()
	opts.NumThreads = loadOpts.Threads
	if loadOpts.Device == "cuda" || loadOpts.Device == "gpu" {
		opts.UseGPU = true
	}

	// Session erstellen (via session.go)
	sessOpts := SessionOptions{
		InputName:   opts.InputName,
		OutputName:  opts.OutputName,
		NumThreads:  opts.NumThreads,
		UseGPU:      opts.UseGPU,
		GPUDeviceID: loadOpts.MainGPU,
	}

	session, err := CreateSession(modelPath, sessOpts)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrSessionCreate, err)
	}

	// Bildgroesse dynamisch aus Modell lesen (Fallback: 224)
	imageSize := session.GetImageSize()
	opts.ImageSize = imageSize

	return &OnnxEncoder{
		session: session,
		info: vision.ModelInfo{
			Name:         "nomic-embed-vision-onnx",
			Type:         "onnx",
			EmbeddingDim: opts.EmbeddingDim,
			ImageSize:    imageSize,
		},
		opts:   opts,
		closed: false,
	}, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode konvertiert ein Bild zu einem 768-dim Embedding-Vektor.
func (e *OnnxEncoder) Encode(imageData []byte) ([]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed {
		return nil, ErrAlreadyClosed
	}
	if len(imageData) == 0 {
		return nil, ErrInvalidInput
	}

	return e.encodeInternal(imageData)
}

// encodeInternal fuehrt die Inference durch (ohne Lock)
func (e *OnnxEncoder) encodeInternal(imageData []byte) ([]float32, error) {
	// Preprocessing (via preprocess.go)
	inputData, err := PreprocessFromBytes(imageData, e.opts.ImageSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrPreprocess, err)
	}

	// Inference (via session.go)
	result, err := e.session.RunInference(inputData, e.opts.EmbeddingDim, e.opts.ImageSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInference, err)
	}

	return result, nil
}

// ============================================================================
// VisionEncoder Interface - EncodeBatch
// ============================================================================

// EncodeBatch konvertiert mehrere Bilder zu Embedding-Vektoren.
func (e *OnnxEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed {
		return nil, ErrAlreadyClosed
	}
	if len(images) == 0 {
		return nil, ErrInvalidInput
	}

	results := make([][]float32, len(images))
	for i, img := range images {
		emb, err := e.encodeInternal(img)
		if err != nil {
			return nil, fmt.Errorf("bild %d: %w", i, err)
		}
		results[i] = emb
	}

	return results, nil
}

// ============================================================================
// VisionEncoder Interface - Close & ModelInfo
// ============================================================================

// Close gibt alle Ressourcen frei
func (e *OnnxEncoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}

	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}

	e.closed = true
	return nil
}

// ModelInfo gibt Metadaten ueber das Modell zurueck
func (e *OnnxEncoder) ModelInfo() vision.ModelInfo {
	return e.info
}

// ============================================================================
// Registry Factory Funktion
// ============================================================================

// OnnxEncoderFactory ist die Factory-Funktion fuer Registry-Registrierung
func OnnxEncoderFactory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewOnnxEncoder(modelPath, opts)
}
