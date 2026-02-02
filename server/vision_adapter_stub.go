//go:build vision && !siglip

// MODUL: vision_adapter_stub
// ZWECK: Stub fuer siglipModelAdapter wenn siglip-Tag nicht gesetzt ist
// INPUT: Keine
// OUTPUT: Stub-Implementierung die Fehler zurueckgibt
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: vision_helpers.go
// HINWEISE: Wird nur mit -tags "vision" (ohne siglip) kompiliert

package server

import (
	"errors"
)

// ============================================================================
// VisionHandler.getModel Stub - Placeholder ohne siglip Support
// ============================================================================

// getModel gibt einen Fehler zurueck da siglip nicht verfuegbar ist.
func (h *VisionHandler) getModel(name string) (VisionEncoderInterface, error) {
	return nil, errors.New("siglip support not compiled in: rebuild with -tags siglip")
}

// ============================================================================
// siglipModelAdapter Stub - Placeholder ohne siglip Support
// ============================================================================

// siglipModelAdapter ist ein Stub wenn siglip nicht verfuegbar ist.
type siglipModelAdapter struct{}

// Encode gibt einen Fehler zurueck da siglip nicht verfuegbar ist.
func (a *siglipModelAdapter) Encode(imageData []byte) ([]float32, error) {
	return nil, errors.New("siglip support not compiled in: rebuild with -tags siglip")
}

// EncodeBatch gibt einen Fehler zurueck da siglip nicht verfuegbar ist.
func (a *siglipModelAdapter) EncodeBatch(images [][]byte) ([][]float32, error) {
	return nil, errors.New("siglip support not compiled in: rebuild with -tags siglip")
}
