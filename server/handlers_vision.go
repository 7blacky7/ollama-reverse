// MODUL: handlers_vision
// ZWECK: HTTP Handler Implementierungen fuer Vision Embedding Endpoints (encode, batch)
// INPUT: HTTP POST Requests mit Base64-kodierten Bildern
// OUTPUT: JSON Responses mit Embedding-Vektoren
// NEBENEFFEKTE: Dekodiert Base64-Bilder, generiert Embeddings ueber Encoder
// ABHAENGIGKEITEN: routes_vision (intern), encoding/base64, net/http (stdlib)
// HINWEISE: Similarity-Handler sind in handlers_vision_similarity.go ausgelagert

package server

import (
	"encoding/base64"
	"fmt"
	"net/http"
)

// ============================================================================
// Fehler-Hilfsfunktionen
// ============================================================================

// errModelNotLoaded erstellt einen Fehler fuer nicht geladene Modelle
func errModelNotLoaded(name string) error {
	return fmt.Errorf("model not loaded: %s", name)
}

// ============================================================================
// POST /api/vision/encode - Einzelbild encodieren
// ============================================================================

// HandleEncode verarbeitet POST /api/vision/encode
func (h *VisionHandler) HandleEncode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionEncodeRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateEncodeRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	embedding, err := h.encodeImage(req.Model, req.Image)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "ENCODING_ERROR")
		return
	}

	h.writeVisionJSON(w, http.StatusOK, VisionEncodeResponse{
		Embedding:  embedding,
		Dimensions: len(embedding),
		Model:      req.Model,
	})
}

// validateEncodeRequest validiert einen Encode-Request
func (h *VisionHandler) validateEncodeRequest(req VisionEncodeRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if req.Image == "" {
		return fmt.Errorf("image is required")
	}
	return nil
}

// encodeImage dekodiert Base64 und generiert Embedding
func (h *VisionHandler) encodeImage(modelName, imageBase64 string) ([]float32, error) {
	imageData, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		return nil, fmt.Errorf("invalid base64: %v", err)
	}

	encoder, err := h.getModel(modelName)
	if err != nil {
		return nil, err
	}

	return encoder.Encode(imageData)
}

// ============================================================================
// POST /api/vision/batch - Mehrere Bilder encodieren
// ============================================================================

// HandleBatch verarbeitet POST /api/vision/batch
func (h *VisionHandler) HandleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionBatchRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateBatchRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	embeddings, err := h.encodeBatch(req.Model, req.Images)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "ENCODING_ERROR")
		return
	}

	dimensions := 0
	if len(embeddings) > 0 {
		dimensions = len(embeddings[0])
	}

	h.writeVisionJSON(w, http.StatusOK, VisionBatchResponse{
		Embeddings: embeddings,
		Dimensions: dimensions,
		Model:      req.Model,
		Count:      len(embeddings),
	})
}

// validateBatchRequest validiert einen Batch-Request
func (h *VisionHandler) validateBatchRequest(req VisionBatchRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if len(req.Images) == 0 {
		return fmt.Errorf("images are required")
	}
	return nil
}

// encodeBatch dekodiert und encodiert mehrere Bilder
func (h *VisionHandler) encodeBatch(modelName string, imagesBase64 []string) ([][]float32, error) {
	imagesData := make([][]byte, len(imagesBase64))
	for i, img := range imagesBase64 {
		data, err := base64.StdEncoding.DecodeString(img)
		if err != nil {
			return nil, fmt.Errorf("invalid base64 at index %d: %v", i, err)
		}
		imagesData[i] = data
	}

	encoder, err := h.getModel(modelName)
	if err != nil {
		return nil, err
	}

	return encoder.EncodeBatch(imagesData)
}
