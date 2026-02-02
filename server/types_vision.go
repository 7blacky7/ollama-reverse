//go:build vision

// MODUL: types_vision
// ZWECK: REST API Types fuer generische Vision Encoder (CLIP, SigLIP, etc.)
// INPUT: Keine (Type-Definitionen)
// OUTPUT: Strukturierte Request/Response Types
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Verwendet fuer /api/vision/* Endpoints
// BUILD-TAG: Kompiliert mit -tags vision
//
// WICHTIG: Modell-Verwaltungs-Types sind in routes_vision_models.go definiert

package server

// ============================================================================
// Vision Encoder Request Types
// ============================================================================

// VisionEncodeRequest - Anfrage fuer Einzelbild-Encoding.
// Endpoint: POST /api/vision/encode
type VisionEncodeRequest struct {
	// Model ist der Encoder-Name (z.B. "clip", "siglip", "blip")
	Model string `json:"model"`

	// Image ist das Base64-kodierte Bild
	Image string `json:"image"`

	// Options sind optionale Encoding-Parameter
	Options *VisionEncodeOptions `json:"options,omitempty"`
}

// VisionEncodeOptions - Optionale Parameter fuer Encoding.
type VisionEncodeOptions struct {
	// Normalize gibt an, ob das Embedding normalisiert werden soll
	Normalize bool `json:"normalize,omitempty"`

	// Precision ist die gewuenschte Praezision (float32, float16)
	Precision string `json:"precision,omitempty"`
}

// VisionBatchRequest - Anfrage fuer Batch-Encoding.
// Endpoint: POST /api/vision/batch
type VisionBatchRequest struct {
	// Model ist der Encoder-Name
	Model string `json:"model"`

	// Images ist die Liste von Base64-kodierten Bildern
	Images []string `json:"images"`

	// Options sind optionale Encoding-Parameter
	Options *VisionEncodeOptions `json:"options,omitempty"`
}

// VisionSimilarityRequest - Anfrage fuer Aehnlichkeitsberechnung.
// Endpoint: POST /api/vision/similarity
type VisionSimilarityRequest struct {
	// Model ist der Encoder-Name
	Model string `json:"model"`

	// Image1 ist das erste Base64-kodierte Bild
	Image1 string `json:"image1"`

	// Image2 ist das zweite Base64-kodierte Bild
	Image2 string `json:"image2"`
}

// ============================================================================
// Vision Encoder Response Types
// ============================================================================

// VisionEncodeResponse - Antwort mit Embedding.
type VisionEncodeResponse struct {
	// Embedding ist das generierte Embedding als float32 Array
	Embedding []float32 `json:"embedding"`

	// Dimensions ist die Groesse des Embeddings
	Dimensions int `json:"dimensions"`

	// Model ist der verwendete Encoder-Name
	Model string `json:"model"`

	// ProcessingTimeMs ist die Verarbeitungszeit in Millisekunden
	ProcessingTimeMs int64 `json:"processing_time_ms,omitempty"`
}

// VisionBatchResponse - Antwort mit mehreren Embeddings.
type VisionBatchResponse struct {
	// Embeddings sind die generierten Embeddings als float32 Arrays
	Embeddings [][]float32 `json:"embeddings"`

	// Dimensions ist die Groesse jedes Embeddings
	Dimensions int `json:"dimensions"`

	// Model ist der verwendete Encoder-Name
	Model string `json:"model"`

	// Count ist die Anzahl der generierten Embeddings
	Count int `json:"count"`

	// ProcessingTimeMs ist die Verarbeitungszeit in Millisekunden
	ProcessingTimeMs int64 `json:"processing_time_ms,omitempty"`
}

// VisionSimilarityResponse - Antwort mit Aehnlichkeitswert.
type VisionSimilarityResponse struct {
	// Similarity ist der Cosine-Similarity-Wert (0.0 - 1.0)
	Similarity float32 `json:"similarity"`

	// Model ist der verwendete Encoder-Name
	Model string `json:"model"`

	// ProcessingTimeMs ist die Verarbeitungszeit in Millisekunden
	ProcessingTimeMs int64 `json:"processing_time_ms,omitempty"`
}
