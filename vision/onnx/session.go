//go:build vision && cgo

// MODUL: onnx/session
// ZWECK: ONNX Runtime Session Management - Erstellen, Konfigurieren, Ausfuehren
// INPUT: Modell-Pfad (.onnx), Session-Optionen, Input-Tensoren
// OUTPUT: Session-Handle, Output-Tensoren
// NEBENEFFEKTE: Alloziert ONNX Runtime Ressourcen, GPU Memory
// ABHAENGIGKEITEN: onnxruntime_go
// HINWEISE: Thread-sicher, Destroy() MUSS aufgerufen werden

package onnx

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ============================================================================
// Runtime Initialisierung (Singleton)
// ============================================================================

var (
	runtimeInitOnce sync.Once
	runtimeInitErr  error
)

// InitRuntime initialisiert die ONNX Runtime einmalig.
// Wird automatisch beim ersten Session-Erstellen aufgerufen.
func InitRuntime() error {
	runtimeInitOnce.Do(func() {
		runtimeInitErr = ort.InitializeEnvironment()
	})
	return runtimeInitErr
}

// DestroyRuntime gibt die ONNX Runtime frei.
// Sollte am Programmende aufgerufen werden.
func DestroyRuntime() error {
	return ort.DestroyEnvironment()
}

// ============================================================================
// Session Struktur
// ============================================================================

// Session verwaltet eine ONNX Runtime Inference Session.
type Session struct {
	inner     *ort.DynamicAdvancedSession
	inputName string
	outputName string
	opts      SessionOptions
}

// SessionOptions konfiguriert die ONNX Session
type SessionOptions struct {
	// InputName ist der ONNX Input-Tensor Name
	InputName string

	// OutputName ist der ONNX Output-Tensor Name
	OutputName string

	// NumThreads fuer Intra-Op Parallelisierung (0 = auto)
	NumThreads int

	// UseGPU aktiviert CUDA Execution Provider
	UseGPU bool

	// GPUDeviceID ist die GPU Index (Standard: 0)
	GPUDeviceID int
}

// DefaultSessionOptions gibt Standard-Optionen zurueck
func DefaultSessionOptions() SessionOptions {
	return SessionOptions{
		InputName:   DefaultInputName,
		OutputName:  DefaultOutputName,
		NumThreads:  0,
		UseGPU:      false,
		GPUDeviceID: 0,
	}
}

// ============================================================================
// Session Konstruktor
// ============================================================================

// CreateSession erstellt eine neue ONNX Inference Session.
func CreateSession(modelPath string, opts SessionOptions) (*Session, error) {
	// Runtime initialisieren falls noetig
	if err := InitRuntime(); err != nil {
		return nil, fmt.Errorf("runtime init: %w", err)
	}

	// Session-Optionen konfigurieren
	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer sessOpts.Destroy()

	// Thread-Anzahl setzen
	if opts.NumThreads > 0 {
		if err := sessOpts.SetIntraOpNumThreads(opts.NumThreads); err != nil {
			return nil, fmt.Errorf("threads setzen: %w", err)
		}
	}

	// GPU aktivieren wenn gewuenscht
	if opts.UseGPU {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			_ = cudaOpts.Update(map[string]string{
				"device_id": fmt.Sprintf("%d", opts.GPUDeviceID),
			})
			_ = sessOpts.AppendExecutionProviderCUDA(cudaOpts)
			cudaOpts.Destroy()
		}
		// Bei Fehler: Fallback auf CPU (kein Error)
	}

	// Session erstellen
	inner, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{opts.InputName},
		[]string{opts.OutputName},
		sessOpts,
	)
	if err != nil {
		return nil, fmt.Errorf("session erstellen: %w", err)
	}

	return &Session{
		inner:      inner,
		inputName:  opts.InputName,
		outputName: opts.OutputName,
		opts:       opts,
	}, nil
}

// ============================================================================
// Session Methoden
// ============================================================================

// RunInference fuehrt Inference mit dem gegebenen Input durch.
// input: Preprocessed Bild-Daten im NCHW Format [1, 3, H, W]
// embeddingDim: Erwartete Output-Dimension
// imageSize: Bild-Groesse (H = W)
// Rueckgabe: Embedding-Vektor der Groesse embeddingDim
func (s *Session) RunInference(input []float32, embeddingDim, imageSize int) ([]float32, error) {
	// Input Tensor erstellen
	inputShape := ort.Shape{1, 3, int64(imageSize), int64(imageSize)}
	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Output Tensor vorbereiten
	outputShape := ort.Shape{1, int64(embeddingDim)}
	outputData := make([]float32, embeddingDim)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Inference ausfuehren
	err = s.inner.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	// Ergebnis kopieren
	result := make([]float32, embeddingDim)
	copy(result, outputTensor.GetData())

	return result, nil
}

// Destroy gibt alle Session-Ressourcen frei
func (s *Session) Destroy() {
	if s.inner != nil {
		s.inner.Destroy()
		s.inner = nil
	}
}
