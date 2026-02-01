// MODUL: routes_vision
// ZWECK: REST API Handler fuer Vision Embedding Endpoints (encode, batch, similarity)
// INPUT: HTTP Requests mit Base64-Bildern und Modell-Namen
// OUTPUT: JSON Responses mit Embeddings und Similarity-Scores
// NEBENEFFEKTE: Laedt Modelle bei Bedarf, schreibt HTTP Responses
// ABHAENGIGKEITEN: vision (intern), encoding/json, net/http, sync (stdlib)
// HINWEISE: Thread-sicher durch RWMutex, Modelle werden gecacht

package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"sync"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Request/Response Types fuer Vision API
// ============================================================================

// VisionEncodeRequest - Request fuer Einzelbild-Encoding
type VisionEncodeRequest struct {
	Model string `json:"model"` // Modell-Name: siglip, clip, nomic, etc.
	Image string `json:"image"` // Base64-kodiertes Bild
}

// VisionEncodeResponse - Response fuer Einzelbild-Encoding
type VisionEncodeResponse struct {
	Embedding  []float32 `json:"embedding"`  // Embedding-Vektor
	Dimensions int       `json:"dimensions"` // Embedding-Dimension
	Model      string    `json:"model"`      // Verwendetes Modell
}

// VisionBatchRequest - Request fuer Batch-Encoding
type VisionBatchRequest struct {
	Model  string   `json:"model"`  // Modell-Name
	Images []string `json:"images"` // Base64-kodierte Bilder
}

// VisionBatchResponse - Response fuer Batch-Encoding
type VisionBatchResponse struct {
	Embeddings [][]float32 `json:"embeddings"` // Embedding-Vektoren
	Dimensions int         `json:"dimensions"` // Embedding-Dimension
	Model      string      `json:"model"`      // Verwendetes Modell
	Count      int         `json:"count"`      // Anzahl Embeddings
}

// VisionSimilarityRequest - Request fuer Similarity-Berechnung
type VisionSimilarityRequest struct {
	Model  string `json:"model"`  // Modell-Name
	Image1 string `json:"image1"` // Erstes Bild (Base64)
	Image2 string `json:"image2"` // Zweites Bild (Base64)
}

// VisionSimilarityResponse - Response fuer Similarity-Berechnung
type VisionSimilarityResponse struct {
	Similarity float32 `json:"similarity"` // Cosine Similarity (0.0 - 1.0)
	Model      string  `json:"model"`      // Verwendetes Modell
}

// VisionErrorResponse - Fehler-Response
type VisionErrorResponse struct {
	Error string `json:"error"` // Fehlermeldung
	Code  string `json:"code"`  // Fehler-Code
}

// ============================================================================
// VisionHandler - Zentrale Handler-Struktur
// ============================================================================

// VisionHandler verwaltet Vision Embedding Endpoints
type VisionHandler struct {
	registry *vision.Registry                // Registry fuer Encoder-Factories
	models   map[string]vision.VisionEncoder // Geladene Modelle (gecacht)
	paths    map[string]string               // Modell-Pfade
	mu       sync.RWMutex                    // Thread-Sicherheit
}

// NewVisionHandler erstellt einen neuen VisionHandler
func NewVisionHandler() *VisionHandler {
	return &VisionHandler{
		registry: vision.DefaultRegistry,
		models:   make(map[string]vision.VisionEncoder),
		paths:    make(map[string]string),
	}
}

// Close schliesst alle geladenen Modelle
func (h *VisionHandler) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	var firstErr error
	for name, encoder := range h.models {
		if err := encoder.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(h.models, name)
	}
	return firstErr
}

// ============================================================================
// Interne Hilfsfunktionen
// ============================================================================

// getModel holt ein Modell aus dem Cache oder gibt Fehler zurueck
func (h *VisionHandler) getModel(name string) (vision.VisionEncoder, error) {
	h.mu.RLock()
	encoder, exists := h.models[name]
	h.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model not loaded: %s", name)
	}
	return encoder, nil
}

// writeVisionJSON schreibt eine JSON-Response
func (h *VisionHandler) writeVisionJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeVisionError schreibt eine Fehler-Response
func (h *VisionHandler) writeVisionError(w http.ResponseWriter, status int, msg, code string) {
	h.writeVisionJSON(w, status, VisionErrorResponse{Error: msg, Code: code})
}

// decodeVisionJSON dekodiert JSON aus dem Request-Body
func decodeVisionJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

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
	// Newton-Raphson Iteration
	z := x / 2
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
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

	// Validierung
	if err := h.validateEncodeRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Bild dekodieren und encodieren
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

	// Validierung
	if err := h.validateBatchRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Batch encodieren
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

	// Validierung
	if err := h.validateSimilarityRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Similarity berechnen
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
// Erweiterter Similarity Handler mit Kandidaten-Liste
// ============================================================================

// VisionSimilarityBatchRequest - Request fuer Batch-Similarity
type VisionSimilarityBatchRequest struct {
	Model      string   `json:"model"`      // Modell-Name
	Query      string   `json:"query"`      // Query-Bild (Base64)
	Candidates []string `json:"candidates"` // Kandidaten-Bilder (Base64)
	TopK       int      `json:"top_k"`      // Anzahl Top-Ergebnisse (optional)
}

// VisionSimilarityBatchResponse - Response fuer Batch-Similarity
type VisionSimilarityBatchResponse struct {
	Results []VisionSimilarityResult `json:"results"` // Sortierte Ergebnisse
	Model   string                   `json:"model"`   // Verwendetes Modell
	Count   int                      `json:"count"`   // Anzahl Ergebnisse
}

// VisionSimilarityResult - Einzelnes Similarity-Ergebnis
type VisionSimilarityResult struct {
	Index int     `json:"index"` // Index des Kandidaten
	Score float32 `json:"score"` // Similarity-Score
}

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

	// Validierung
	if err := h.validateSimilarityBatchRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Similarity berechnen
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
	// Query-Embedding generieren
	queryEmb, err := h.encodeImage(req.Model, req.Query)
	if err != nil {
		return nil, fmt.Errorf("query encoding failed: %v", err)
	}

	// Kandidaten-Embeddings generieren
	candEmbs, err := h.encodeBatch(req.Model, req.Candidates)
	if err != nil {
		return nil, fmt.Errorf("candidates encoding failed: %v", err)
	}

	// Similarity berechnen
	results := make([]VisionSimilarityResult, len(candEmbs))
	for i, candEmb := range candEmbs {
		results[i] = VisionSimilarityResult{
			Index: i,
			Score: cosineSimilarity(queryEmb, candEmb),
		}
	}

	// Nach Score sortieren (absteigend)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Top-K begrenzen
	topK := req.TopK
	if topK <= 0 || topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

// ============================================================================
// Router Registration
// ============================================================================

// RegisterVisionRoutes registriert alle Vision-Routes auf einem http.ServeMux
func (h *VisionHandler) RegisterVisionRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/vision/encode", h.HandleEncode)
	mux.HandleFunc("/api/vision/batch", h.HandleBatch)
	mux.HandleFunc("/api/vision/similarity", h.HandleSimilarity)
	mux.HandleFunc("/api/vision/similarity/batch", h.HandleSimilarityBatch)
}
