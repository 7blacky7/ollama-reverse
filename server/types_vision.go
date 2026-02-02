
// MODUL: types_vision
// ZWECK: REST API Types fuer generische Vision Encoder (CLIP, SigLIP, etc.)
// INPUT: Keine (Type-Definitionen)
// OUTPUT: Strukturierte Request/Response Types
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Verwendet fuer /api/vision/* Endpoints
// BUILD-TAG: Nur mit -tags vision_types (Types bereits in routes_*.go definiert)

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

// VisionLoadRequest - Anfrage zum Laden eines Modells.
// Endpoint: POST /api/vision/load
type VisionLoadRequest struct {
	// Model ist der Encoder-Typ (z.B. "clip", "siglip")
	Model string `json:"model"`

	// Path ist der Pfad zur GGUF-Datei
	Path string `json:"path"`

	// Backend ist das gewuenschte Compute-Backend (optional)
	Backend string `json:"backend,omitempty"`
}

// VisionUnloadRequest - Anfrage zum Entladen eines Modells.
// Endpoint: POST /api/vision/unload
type VisionUnloadRequest struct {
	// Model ist der Name des zu entladenden Modells
	Model string `json:"model"`
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

// VisionModelInfo - Informationen zu einem Vision-Encoder.
type VisionModelInfo struct {
	// Name ist der Modell-Name
	Name string `json:"name"`

	// Path ist der Pfad zur Modell-Datei
	Path string `json:"path"`

	// EmbeddingDim ist die Dimension der Embeddings
	EmbeddingDim int `json:"embedding_dim"`

	// ImageSize ist die erwartete Bildgroesse in Pixeln
	ImageSize int `json:"image_size,omitempty"`

	// Type ist der Encoder-Typ (z.B. "clip-vit-b", "siglip-so400m")
	Type string `json:"type,omitempty"`

	// Backend ist das verwendete Compute-Backend
	Backend string `json:"backend,omitempty"`

	// SizeBytes ist die Groesse des Modells in Bytes
	SizeBytes int64 `json:"size_bytes,omitempty"`
}

// VisionModelsResponse - Liste verfuegbarer Encoder.
// Endpoint: GET /api/vision/models
type VisionModelsResponse struct {
	// Models ist die Liste der verfuegbaren Modell-Namen
	Models []string `json:"models"`

	// LoadedModels ist eine Map von Namen zu Pfaden (nur geladene Modelle)
	LoadedModels map[string]VisionModelInfo `json:"loaded_models"`
}

// VisionLoadResponse - Antwort nach dem Laden eines Modells.
type VisionLoadResponse struct {
	// Success gibt an, ob das Laden erfolgreich war
	Success bool `json:"success"`

	// Model ist der Name des geladenen Modells
	Model string `json:"model"`

	// Info enthaelt Details zum geladenen Modell
	Info *VisionModelInfo `json:"info,omitempty"`

	// Message ist eine optionale Nachricht
	Message string `json:"message,omitempty"`
}

// VisionUnloadResponse - Antwort nach dem Entladen eines Modells.
type VisionUnloadResponse struct {
	// Success gibt an, ob das Entladen erfolgreich war
	Success bool `json:"success"`

	// Model ist der Name des entladenen Modells
	Model string `json:"model"`

	// Message ist eine optionale Nachricht
	Message string `json:"message,omitempty"`
}
