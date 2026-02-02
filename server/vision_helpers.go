//go:build vision

// MODUL: vision_helpers
// ZWECK: Helper-Methoden fuer VisionHandler (Response-Writer, Validierung, Model-Zugriff)
// INPUT: HTTP ResponseWriter, Requests, Model-Namen
// OUTPUT: JSON Responses, validierte Daten, Encoder-Interfaces
// NEBENEFFEKTE: Schreibt HTTP Responses
// ABHAENGIGKEITEN: encoding/json, net/http, errors (stdlib)
// HINWEISE: Abstrahiert siglip.Model hinter VisionEncoderInterface

package server

import (
	"encoding/json"
	"net/http"
)

// ============================================================================
// VisionEncoderInterface - Abstraction fuer verschiedene Vision Encoder
// ============================================================================

// VisionEncoderInterface definiert das Interface fuer Vision Encoder.
// Wird von siglip.Model und vision.VisionEncoder implementiert.
type VisionEncoderInterface interface {
	// Encode generiert ein Embedding fuer ein einzelnes Bild
	Encode(imageData []byte) ([]float32, error)

	// EncodeBatch generiert Embeddings fuer mehrere Bilder
	EncodeBatch(images [][]byte) ([][]float32, error)
}

// ============================================================================
// HTTP Response Helper fuer VisionHandler
// ============================================================================

// writeVisionError schreibt einen JSON-Fehler Response.
// Implementiert als Methode auf VisionHandler fuer konsistente API.
func (h *VisionHandler) writeVisionError(w http.ResponseWriter, status int, message, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(VisionErrorResponse{
		Error: message,
		Code:  code,
	})
}

// writeVisionJSON schreibt einen JSON-Success Response.
func (h *VisionHandler) writeVisionJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// ============================================================================
// JSON Decoding Helper
// ============================================================================

// decodeVisionJSON dekodiert JSON aus einem HTTP Request Body.
// Standalone-Funktion fuer Wiederverwendbarkeit.
func decodeVisionJSON(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}
