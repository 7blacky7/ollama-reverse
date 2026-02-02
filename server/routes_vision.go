//go:build vision

// MODUL: routes_vision
// ZWECK: Route-Registrierung fuer Vision API als VisionHandler-Methode
// INPUT: http.ServeMux
// OUTPUT: Konfigurierter HTTP-Router
// NEBENEFFEKTE: Registriert HTTP-Routen
// ABHAENGIGKEITEN: router_vision (VisionHandler)
// HINWEISE: Types sind in types_vision.go definiert

package server

import (
	"net/http"
)

// ============================================================================
// Route Registration als Methode (fuer RegisterAllVisionRoutes)
// ============================================================================

// RegisterVisionRoutes registriert die Encoding-Routes als Handler-Methode.
// Ermoeglicht Kombination mit RegisterVisionModelRoutes.
func (h *VisionHandler) RegisterVisionRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/vision/encode", h.HandleEncode)
	mux.HandleFunc("/api/vision/batch", h.HandleBatch)
	mux.HandleFunc("/api/vision/similarity", h.HandleSimilarity)
	mux.HandleFunc("/api/vision/similarity/batch", h.HandleSimilarityBatch)
}

// ============================================================================
// Additional Types fuer Similarity Batch
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

// VisionErrorResponse - Fehler-Response (Legacy-Kompatibilitaet)
type VisionErrorResponse struct {
	Error string `json:"error"` // Fehlermeldung
	Code  string `json:"code"`  // Fehler-Code
}
