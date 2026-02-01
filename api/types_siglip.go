// Package api definiert die Request/Response Types fuer die SigLIP REST API.
package api

// ============================================================================
// Embed Endpoint Types
// ============================================================================

// EmbedImageRequest repraesentiert einen Request fuer ein einzelnes Image-Embedding.
//
// POST /api/embed/image
type EmbedImageRequest struct {
	// Model ist der Name des SigLIP-Modells (z.B. "siglip-vit-b")
	Model string `json:"model"`

	// Image ist das Base64-kodierte Bild
	Image string `json:"image"`
}

// EmbedImageResponse repraesentiert die Response fuer ein einzelnes Image-Embedding.
type EmbedImageResponse struct {
	// Embedding ist das generierte Embedding als float32 Array
	Embedding []float32 `json:"embedding"`

	// Model ist der Name des verwendeten Modells
	Model string `json:"model"`

	// Dimension ist die Groesse des Embeddings
	Dimension int `json:"dimension,omitempty"`
}

// ============================================================================
// Batch Embed Endpoint Types
// ============================================================================

// EmbedBatchRequest repraesentiert einen Request fuer Batch Image-Embeddings.
//
// POST /api/embed/batch
type EmbedBatchRequest struct {
	// Model ist der Name des SigLIP-Modells
	Model string `json:"model"`

	// Images ist eine Liste von Base64-kodierten Bildern
	Images []string `json:"images"`
}

// EmbedBatchResponse repraesentiert die Response fuer Batch Image-Embeddings.
type EmbedBatchResponse struct {
	// Embeddings sind die generierten Embeddings als float32 Arrays
	Embeddings [][]float32 `json:"embeddings"`

	// Model ist der Name des verwendeten Modells
	Model string `json:"model"`

	// Count ist die Anzahl der generierten Embeddings
	Count int `json:"count,omitempty"`

	// Dimension ist die Groesse jedes Embeddings
	Dimension int `json:"dimension,omitempty"`
}

// ============================================================================
// Similarity Endpoint Types
// ============================================================================

// SimilarityRequest repraesentiert einen Request fuer Image-Similarity-Suche.
//
// POST /api/similarity
type SimilarityRequest struct {
	// Model ist der Name des SigLIP-Modells
	Model string `json:"model"`

	// Image ist das Query-Bild (Base64-kodiert)
	Image string `json:"image"`

	// Candidates sind die Kandidaten-Bilder (Base64-kodiert)
	Candidates []string `json:"candidates"`

	// TopK ist die Anzahl der zurueckzugebenden Top-Ergebnisse (optional, default: alle)
	TopK int `json:"top_k,omitempty"`
}

// SimilarityResult repraesentiert ein einzelnes Similarity-Ergebnis.
type SimilarityResult struct {
	// Index ist der Index des Kandidaten-Bildes
	Index int `json:"index"`

	// Score ist die Cosine Similarity (0.0 - 1.0)
	Score float32 `json:"score"`
}

// SimilarityResponse repraesentiert die Response fuer Image-Similarity-Suche.
type SimilarityResponse struct {
	// Results sind die Similarity-Ergebnisse (sortiert nach Score, absteigend)
	Results []SimilarityResult `json:"results"`

	// Model ist der Name des verwendeten Modells
	Model string `json:"model"`

	// Count ist die Anzahl der Ergebnisse
	Count int `json:"count,omitempty"`
}

// ============================================================================
// Models Endpoint Types
// ============================================================================

// ModelInfo repraesentiert Informationen zu einem geladenen SigLIP-Modell.
type ModelInfo struct {
	// Name ist der Name des Modells
	Name string `json:"name"`

	// Size ist die Groesse des Modells in Bytes
	Size int64 `json:"size"`

	// EmbeddingDim ist die Dimension der Embeddings
	EmbeddingDim int `json:"embedding_dim"`

	// ImageSize ist die erwartete Bildgroesse (Pixels)
	ImageSize int `json:"image_size,omitempty"`

	// Type ist der Modell-Typ (z.B. "ViT-B/16")
	Type string `json:"type,omitempty"`

	// Backend ist das verwendete Compute-Backend
	Backend string `json:"backend,omitempty"`
}

// ListModelsResponse repraesentiert die Response fuer /api/siglip/models.
type ListModelsResponse struct {
	// Models ist die Liste der verfuegbaren Modelle
	Models []ModelInfo `json:"models"`
}

// ============================================================================
// Error Response Types
// ============================================================================

// ErrorResponse repraesentiert eine Fehler-Antwort.
type ErrorResponse struct {
	// Error ist die Fehlermeldung
	Error string `json:"error"`

	// Code ist ein optionaler Fehler-Code
	Code string `json:"code,omitempty"`

	// Details enthaelt optionale zusaetzliche Informationen
	Details map[string]interface{} `json:"details,omitempty"`
}

// ============================================================================
// Health/Info Endpoint Types
// ============================================================================

// SigLIPInfoResponse repraesentiert Informationen ueber die SigLIP-Komponente.
type SigLIPInfoResponse struct {
	// Version ist die SigLIP-Version
	Version string `json:"version"`

	// BuildInfo sind Build-Informationen
	BuildInfo string `json:"build_info,omitempty"`

	// SystemInfo sind System-Informationen
	SystemInfo string `json:"system_info,omitempty"`

	// AvailableBackends ist die Liste verfuegbarer Backends
	AvailableBackends []string `json:"available_backends,omitempty"`
}
