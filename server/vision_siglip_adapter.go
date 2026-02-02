//go:build vision && siglip

// MODUL: vision_siglip_adapter
// ZWECK: Adapter fuer siglip.Model zu VisionEncoderInterface
// INPUT: siglip.Model
// OUTPUT: VisionEncoderInterface Implementation
// NEBENEFFEKTE: Delegiert an siglip.Model
// ABHAENGIGKEITEN: siglip (intern), vision_helpers.go
// HINWEISE: Wird nur mit -tags "vision,siglip" kompiliert

package server

import (
	"errors"

	"github.com/ollama/ollama/siglip"
)

// ============================================================================
// VisionHandler.getModel - Model-Zugriff mit siglip Support
// ============================================================================

// getModel holt einen Encoder aus dem ModelManager.
// Gibt ein VisionEncoderInterface zurueck, das von siglip.Model implementiert wird.
func (h *VisionHandler) getModel(name string) (VisionEncoderInterface, error) {
	if h.manager == nil {
		return nil, errors.New("vision manager not initialized")
	}

	model, err := h.manager.GetModel(name)
	if err != nil {
		return nil, err
	}

	// Wrap siglip.Model in einen Adapter der VisionEncoderInterface implementiert
	return &siglipModelAdapter{model: model}, nil
}

// ============================================================================
// siglipModelAdapter - Wrapper fuer siglip.Model
// ============================================================================

// siglipModelAdapter adaptiert siglip.Model fuer das VisionEncoderInterface.
// Konvertiert *siglip.Embedding zu []float32.
type siglipModelAdapter struct {
	model *siglip.Model
}

// Encode generiert ein Embedding fuer ein einzelnes Bild.
// Implementiert VisionEncoderInterface.Encode().
func (a *siglipModelAdapter) Encode(imageData []byte) ([]float32, error) {
	embedding, err := a.model.Encode(imageData)
	if err != nil {
		return nil, err
	}
	return embedding.ToFloat32(), nil
}

// EncodeBatch generiert Embeddings fuer mehrere Bilder.
// Implementiert VisionEncoderInterface.EncodeBatch().
func (a *siglipModelAdapter) EncodeBatch(images [][]byte) ([][]float32, error) {
	embeddings, err := a.model.EncodeBatch(images)
	if err != nil {
		return nil, err
	}

	// []*siglip.Embedding zu [][]float32 konvertieren
	result := make([][]float32, len(embeddings))
	for i, emb := range embeddings {
		if emb != nil {
			result[i] = emb.ToFloat32()
		}
	}
	return result, nil
}
