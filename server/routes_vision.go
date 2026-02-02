// MODUL: routes_vision
// ZWECK: REST API Definitionen und VisionHandler-Struktur fuer Vision Embedding Endpoints
// INPUT: HTTP Requests mit Base64-Bildern und Modell-Namen
// OUTPUT: JSON Responses mit Embeddings und Similarity-Scores
// NEBENEFFEKTE: Verwaltet geladene Modelle im Cache
// ABHAENGIGKEITEN: vision (intern), handlers_vision (intern), encoding/json, net/http, sync (stdlib)
// HINWEISE: Handler-Implementierungen sind in handlers_vision.go ausgelagert

package server

import (
	"encoding/json"
	"net/http"
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
		return nil, errModelNotLoaded(name)
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
