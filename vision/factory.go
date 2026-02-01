// MODUL: factory
// ZWECK: Factory Pattern fuer Vision Encoder mit Auto-Detection
// INPUT: Modell-Typ, Modell-Pfad, Options
// OUTPUT: VisionEncoder Interface Implementation
// NEBENEFFEKTE: Laedt Modell-Dateien, alloziert Speicher
// ABHAENGIGKEITEN: registry.go (Registry, EncoderFactory), gguf.go
// HINWEISE: Nutzt DefaultRegistry aus registry_global.go

package vision

import (
	"errors"
	"fmt"
	"os"
)

// ============================================================================
// Fehler-Definitionen fuer Factory
// ============================================================================

var (
	ErrUnknownModelType = errors.New("vision: unknown model type")
	ErrModelNotFound    = errors.New("vision: model file not found")
)

// ============================================================================
// ModelConfig - Konfiguration fuer Encoder
// ============================================================================

// ModelConfig enthaelt die vollstaendige Modell-Konfiguration.
type ModelConfig struct {
	Type    string      // Modell-Typ: "siglip", "clip", "nomic", "dinov2"
	Path    string      // Pfad zur GGUF-Datei
	Options LoadOptions // Lade-Optionen
}

// ============================================================================
// VisionEncoder Interface
// ============================================================================

// VisionEncoder ist das zentrale Interface fuer alle Vision Encoder.
type VisionEncoder interface {
	Encode(imageData []byte) ([]float32, error)
	EncodeBatch(images [][]byte) ([][]float32, error)
	Close() error
	ModelInfo() ModelInfo
}

// ModelInfo enthaelt Metadaten ueber ein geladenes Modell.
type ModelInfo struct {
	Name         string // Modell-Name
	Type         string // Modell-Typ
	EmbeddingDim int    // Embedding-Dimension
	ImageSize    int    // Erwartete Bildgroesse
}

// ============================================================================
// EncoderFactory - Factory-Funktion Typ
// ============================================================================

// EncoderFactory ist eine Funktion die einen Encoder erstellt.
// Wird von Registry.Register() verwendet.
type EncoderFactory func(modelPath string, opts LoadOptions) (VisionEncoder, error)

// ============================================================================
// NewEncoder - Hauptfunktion fuer Encoder-Erstellung
// ============================================================================

// NewEncoder erstellt einen Vision Encoder basierend auf Typ und Pfad.
// Verwendet DefaultRegistry zur Factory-Aufloesung.
func NewEncoder(modelType string, modelPath string, opts ...Option) (VisionEncoder, error) {
	// Optionen anwenden
	loadOpts := DefaultLoadOptions()
	loadOpts.Apply(opts...)

	// Optionen validieren
	if err := loadOpts.Validate(); err != nil {
		return nil, err
	}

	// Modell-Datei pruefen
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, ErrModelNotFound
	}

	// Factory aus DefaultRegistry holen
	factory, exists := DefaultRegistry.Get(modelType)
	if !exists {
		return nil, fmt.Errorf("%w: %s", ErrUnknownModelType, modelType)
	}

	return factory(modelPath, loadOpts)
}

// ============================================================================
// NewEncoderFromConfig - Encoder aus Config erstellen
// ============================================================================

// NewEncoderFromConfig erstellt einen Encoder aus einer ModelConfig.
func NewEncoderFromConfig(config ModelConfig) (VisionEncoder, error) {
	return NewEncoder(config.Type, config.Path,
		WithDevice(config.Options.Device),
		WithThreads(config.Options.Threads),
		WithBatchSize(config.Options.BatchSize),
		WithQuantization(config.Options.Quantization),
		WithGPULayers(config.Options.GPULayers),
		WithMainGPU(config.Options.MainGPU),
		WithMmap(config.Options.UseMmap),
		WithMlock(config.Options.UseMlock),
	)
}

// ============================================================================
// AutoDetectEncoder - Modell-Typ aus GGUF erkennen
// ============================================================================

// AutoDetectEncoder erkennt den Modell-Typ aus einer GGUF-Datei.
// Liest den GGUF-Header und sucht nach dem "general.architecture" Key.
func AutoDetectEncoder(modelPath string) (string, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return "", ErrModelNotFound
	}
	defer f.Close()

	return detectModelTypeFromGGUF(f)
}

// ============================================================================
// NewEncoderAuto - Encoder mit automatischer Typ-Erkennung
// ============================================================================

// NewEncoderAuto erkennt den Modell-Typ automatisch und erstellt den Encoder.
func NewEncoderAuto(modelPath string, opts ...Option) (VisionEncoder, error) {
	modelType, err := AutoDetectEncoder(modelPath)
	if err != nil {
		return nil, err
	}

	return NewEncoder(modelType, modelPath, opts...)
}
