// ============================================================================
// MODUL: vision_adapter
// ZWECK: Adapter zur Integration von SigLIP in das vision.VisionEncoder Interface
// INPUT: Modell-Pfad, LoadOptions, Bilddaten ([]byte)
// OUTPUT: VisionEncoder-kompatible Embeddings ([]float32)
// NEBENEFFEKTE: Laedt SigLIP-Modell, alloziert Speicher
// ABHAENGIGKEITEN: siglip/model.go, siglip/embedding.go, vision/factory.go
// HINWEISE: Wrapper-Pattern - delegiert an existierendes Model
// ============================================================================

package siglip

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// SigLIPVisionEncoder - Adapter Struct
// ============================================================================

// SigLIPVisionEncoder adaptiert das SigLIP Model fuer das VisionEncoder Interface.
// Kapselt ein geladenes SigLIP Model und stellt die vision.VisionEncoder API bereit.
type SigLIPVisionEncoder struct {
	model     *Model // Bestehendes SigLIP Model aus model.go
	modelPath string // Pfad zur GGUF-Datei (fuer ModelInfo)
}

// ============================================================================
// Konstruktor - NewSigLIPVisionEncoder
// ============================================================================

// NewSigLIPVisionEncoder erstellt einen neuen SigLIP Encoder aus einem Modell-Pfad.
// Konvertiert vision.LoadOptions zu siglip.Options und laedt das Modell.
func NewSigLIPVisionEncoder(modelPath string, opts vision.LoadOptions) (*SigLIPVisionEncoder, error) {
	// vision.LoadOptions zu siglip.Options konvertieren
	siglipOpts := convertLoadOptions(opts)

	// Modell laden
	model, err := LoadModel(modelPath, siglipOpts...)
	if err != nil {
		return nil, err
	}

	return &SigLIPVisionEncoder{
		model:     model,
		modelPath: modelPath,
	}, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode generiert ein Embedding fuer ein einzelnes Bild.
// Implementiert vision.VisionEncoder.Encode().
func (e *SigLIPVisionEncoder) Encode(imageData []byte) ([]float32, error) {
	// Delegiere an bestehendes Model
	embedding, err := e.model.Encode(imageData)
	if err != nil {
		return nil, err
	}

	// Embedding zu []float32 konvertieren
	return embedding.ToFloat32(), nil
}

// ============================================================================
// VisionEncoder Interface - EncodeBatch
// ============================================================================

// EncodeBatch generiert Embeddings fuer mehrere Bilder.
// Implementiert vision.VisionEncoder.EncodeBatch().
func (e *SigLIPVisionEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
	// Delegiere an bestehendes Model
	embeddings, err := e.model.EncodeBatch(images)
	if err != nil {
		return nil, err
	}

	// []*Embedding zu [][]float32 konvertieren
	return convertEmbeddings(embeddings), nil
}

// ============================================================================
// VisionEncoder Interface - Close
// ============================================================================

// Close gibt das Modell frei.
// Implementiert vision.VisionEncoder.Close().
func (e *SigLIPVisionEncoder) Close() error {
	if e.model != nil {
		return e.model.Close()
	}
	return nil
}

// ============================================================================
// VisionEncoder Interface - ModelInfo
// ============================================================================

// ModelInfo gibt Metadaten ueber das geladene Modell zurueck.
// Implementiert vision.VisionEncoder.ModelInfo().
func (e *SigLIPVisionEncoder) ModelInfo() vision.ModelInfo {
	return vision.ModelInfo{
		Name:         e.model.ModelName(),
		Type:         "siglip",
		EmbeddingDim: e.model.EmbeddingDim(),
		ImageSize:    e.model.ImageSize(),
	}
}

// ============================================================================
// Hilfsfunktionen - Options-Konvertierung
// ============================================================================

// convertLoadOptions konvertiert vision.LoadOptions zu siglip.Option Slice.
func convertLoadOptions(opts vision.LoadOptions) []Option {
	var siglipOpts []Option

	// Backend basierend auf Device setzen
	siglipOpts = append(siglipOpts, WithBackend(deviceToBackend(opts.Device)))

	// Thread-Anzahl
	if opts.Threads > 0 {
		siglipOpts = append(siglipOpts, WithThreads(opts.Threads))
	}

	// Batch-Groesse
	if opts.BatchSize > 0 {
		siglipOpts = append(siglipOpts, WithBatchSize(opts.BatchSize))
	}

	// GPU-Layers
	siglipOpts = append(siglipOpts, WithGPULayers(opts.GPULayers))

	// Main-GPU
	siglipOpts = append(siglipOpts, WithMainGPU(opts.MainGPU))

	// Memory-Mapping
	siglipOpts = append(siglipOpts, WithMmap(opts.UseMmap))

	// Memory-Locking
	siglipOpts = append(siglipOpts, WithMlock(opts.UseMlock))

	return siglipOpts
}

// ============================================================================
// Hilfsfunktionen - Device/Backend Mapping
// ============================================================================

// deviceToBackend konvertiert einen vision.Device String zu siglip.Backend.
func deviceToBackend(device string) Backend {
	switch device {
	case vision.DeviceCUDA:
		return BackendCUDA
	case vision.DeviceMetal:
		return BackendMetal
	default:
		return BackendCPU
	}
}

// ============================================================================
// Hilfsfunktionen - Embedding-Konvertierung
// ============================================================================

// convertEmbeddings konvertiert []*Embedding zu [][]float32.
func convertEmbeddings(embeddings []*Embedding) [][]float32 {
	result := make([][]float32, len(embeddings))
	for i, emb := range embeddings {
		if emb != nil {
			result[i] = emb.ToFloat32()
		}
	}
	return result
}
