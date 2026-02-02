//go:build vision
// MODUL: handlers_vision_similarity
// ZWECK: HTTP Handler fuer Vision Similarity Endpoints (einzel und batch)
// INPUT: HTTP POST Requests mit Base64-kodierten Bildern
// OUTPUT: JSON Responses mit Similarity-Scores
// NEBENEFFEKTE: Generiert Embeddings und berechnet Cosine Similarity
// ABHAENGIGKEITEN: routes_vision (intern), handlers_vision (intern), net/http, sort (stdlib)
// HINWEISE: Nutzt cosineSimilarity und sqrt32 fuer Berechnungen

package server

import (
	"fmt"
	"net/http"
	"sort"
)

// ============================================================================
// Mathematische Hilfsfunktionen
// ============================================================================

// cosineSimilarity berechnet die Cosine Similarity zwischen zwei Vektoren
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (sqrt32(normA) * sqrt32(normB))
}

// sqrt32 berechnet die Quadratwurzel fuer float32
func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// ============================================================================
// POST /api/vision/similarity - Bild-Aehnlichkeit berechnen
// ============================================================================

// HandleSimilarity verarbeitet POST /api/vision/similarity
func (h *VisionHandler) HandleSimilarity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionSimilarityRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateSimilarityRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	similarity, err := h.calculateSimilarity(req)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "SIMILARITY_ERROR")
		return
	}

	h.writeVisionJSON(w, http.StatusOK, VisionSimilarityResponse{
		Similarity: similarity,
		Model:      req.Model,
	})
}

// validateSimilarityRequest validiert einen Similarity-Request
func (h *VisionHandler) validateSimilarityRequest(req VisionSimilarityRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if req.Image1 == "" {
		return fmt.Errorf("image1 is required")
	}
	if req.Image2 == "" {
		return fmt.Errorf("image2 is required")
	}
	return nil
}

// calculateSimilarity berechnet die Similarity zwischen zwei Bildern
func (h *VisionHandler) calculateSimilarity(req VisionSimilarityRequest) (float32, error) {
	emb1, err := h.encodeImage(req.Model, req.Image1)
	if err != nil {
		return 0, fmt.Errorf("image1 encoding failed: %v", err)
	}

	emb2, err := h.encodeImage(req.Model, req.Image2)
	if err != nil {
		return 0, fmt.Errorf("image2 encoding failed: %v", err)
	}

	return cosineSimilarity(emb1, emb2), nil
}

// ============================================================================
// POST /api/vision/similarity/batch - Batch-Similarity berechnen
// ============================================================================

// HandleSimilarityBatch verarbeitet POST /api/vision/similarity/batch
func (h *VisionHandler) HandleSimilarityBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionSimilarityBatchRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateSimilarityBatchRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	results, err := h.calculateSimilarityBatch(req)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "SIMILARITY_ERROR")
		return
	}

	h.writeVisionJSON(w, http.StatusOK, VisionSimilarityBatchResponse{
		Results: results,
		Model:   req.Model,
		Count:   len(results),
	})
}

// validateSimilarityBatchRequest validiert einen Batch-Similarity-Request
func (h *VisionHandler) validateSimilarityBatchRequest(req VisionSimilarityBatchRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if req.Query == "" {
		return fmt.Errorf("query is required")
	}
	if len(req.Candidates) == 0 {
		return fmt.Errorf("candidates are required")
	}
	return nil
}

// calculateSimilarityBatch berechnet Similarity fuer mehrere Kandidaten
func (h *VisionHandler) calculateSimilarityBatch(req VisionSimilarityBatchRequest) ([]VisionSimilarityResult, error) {
	queryEmb, err := h.encodeImage(req.Model, req.Query)
	if err != nil {
		return nil, fmt.Errorf("query encoding failed: %v", err)
	}

	candEmbs, err := h.encodeBatch(req.Model, req.Candidates)
	if err != nil {
		return nil, fmt.Errorf("candidates encoding failed: %v", err)
	}

	results := make([]VisionSimilarityResult, len(candEmbs))
	for i, candEmb := range candEmbs {
		results[i] = VisionSimilarityResult{
			Index: i,
			Score: cosineSimilarity(queryEmb, candEmb),
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	topK := req.TopK
	if topK <= 0 || topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}
