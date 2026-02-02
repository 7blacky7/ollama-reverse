//go:build vision

// MODUL: routes_vision
// ZWECK: REST API Definitionen und VisionHandler-Struktur fuer Vision Embedding Endpoints
// INPUT: HTTP Requests mit Base64-Bildern und Modell-Namen
// OUTPUT: JSON Responses mit Embeddings und Similarity-Scores
// NEBENEFFEKTE: Verwaltet geladene Modelle im Cache
// ABHAENGIGKEITEN: vision (intern), handlers_vision (intern), encoding/json, net/http, sync (stdlib)
// HINWEISE: Handler-Implementierungen sind in handlers_vision.go ausgelagert

package server

// NOTE: VisionHandler and related functions are now in router_vision.go
// This file only contains Request/Response types for the Vision API

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
