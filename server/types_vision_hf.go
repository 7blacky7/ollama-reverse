// MODUL: types_vision_hf
// ZWECK: Request/Response Types fuer HuggingFace Vision Model API
// INPUT: Keine (Type-Definitionen)
// OUTPUT: Strukturierte Request/Response Types fuer HF-Endpoints
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Verwendet fuer /api/vision/**/hf Endpoints

package server

// ============================================================================
// Konstanten fuer HuggingFace Vision API
// ============================================================================

const (
	// HFModelStatusLoaded zeigt an, dass das Modell geladen ist
	HFModelStatusLoaded = "loaded"

	// HFModelStatusCached zeigt an, dass das Modell aus dem Cache geladen wurde
	HFModelStatusCached = "cached"

	// HFModelStatusConverting zeigt an, dass das Modell konvertiert wird
	HFModelStatusConverting = "converting"

	// HFModelStatusDownloading zeigt an, dass das Modell heruntergeladen wird
	HFModelStatusDownloading = "downloading"

	// HFModelStatusError zeigt einen Fehler an
	HFModelStatusError = "error"
)

// Unterstuetzte HuggingFace Encoder-Typen
const (
	// HFEncoderTypeSigLIP ist der SigLIP Vision Encoder
	HFEncoderTypeSigLIP = "siglip"

	// HFEncoderTypeCLIP ist der CLIP Vision Encoder
	HFEncoderTypeCLIP = "clip"

	// HFEncoderTypeOpenCLIP ist der OpenCLIP Vision Encoder
	HFEncoderTypeOpenCLIP = "openclip"

	// HFEncoderTypeBLIP ist der BLIP Vision Encoder
	HFEncoderTypeBLIP = "blip"

	// HFEncoderTypeNomic ist der Nomic Vision Encoder
	HFEncoderTypeNomic = "nomic"
)

// Cache-bezogene Konstanten
const (
	// DefaultCacheDir ist das Standard-Cache-Verzeichnis
	DefaultCacheDir = ".cache/huggingface"

	// MaxCacheSizeBytes ist das maximale Cache-Groesse (10 GB)
	MaxCacheSizeBytes int64 = 10 * 1024 * 1024 * 1024
)

// ============================================================================
// HuggingFace Model Load Request/Response
// ============================================================================

// LoadHFModelRequest - Anfrage zum Laden eines HuggingFace Vision Modells.
// Endpoint: POST /api/vision/load/hf
type LoadHFModelRequest struct {
	// ModelID ist die HuggingFace Model-ID (z.B. "google/siglip-base-patch16-224")
	ModelID string `json:"model_id"`

	// Revision ist optional: Branch, Tag oder Commit-Hash
	Revision string `json:"revision,omitempty"`

	// Force erzwingt das Neuladen auch bei Cache-Hit
	Force bool `json:"force,omitempty"`

	// Backend ist das gewuenschte Compute-Backend (optional)
	Backend string `json:"backend,omitempty"`
}

// LoadHFModelResponse - Antwort nach dem Laden eines HF-Modells.
type LoadHFModelResponse struct {
	// Status ist der aktuelle Lade-Status (loaded, cached, converting, error)
	Status string `json:"status"`

	// EncoderType ist der erkannte Encoder-Typ (siglip, clip, etc.)
	EncoderType string `json:"encoder_type"`

	// ModelPath ist der Pfad zur GGUF-Datei
	ModelPath string `json:"model_path"`

	// CacheHit gibt an, ob das Modell aus dem Cache geladen wurde
	CacheHit bool `json:"cache_hit"`

	// ModelID ist die urspruengliche HuggingFace Model-ID
	ModelID string `json:"model_id"`

	// EmbeddingDim ist die Dimension der Embeddings
	EmbeddingDim int `json:"embedding_dim,omitempty"`

	// Message ist eine optionale Status-Nachricht
	Message string `json:"message,omitempty"`
}

// ============================================================================
// HuggingFace Model Info Types
// ============================================================================

// HFModelInfo - Informationen zu einem HuggingFace Vision Modell.
// Endpoint: GET /api/vision/models/hf
type HFModelInfo struct {
	// ModelID ist die HuggingFace Model-ID
	ModelID string `json:"model_id"`

	// Type ist der Encoder-Typ (siglip, clip, etc.)
	Type string `json:"type"`

	// Description ist eine kurze Beschreibung des Modells
	Description string `json:"description"`

	// Supported gibt an, ob das Modell unterstuetzt wird
	Supported bool `json:"supported"`

	// EmbeddingDim ist die erwartete Embedding-Dimension
	EmbeddingDim int `json:"embedding_dim,omitempty"`

	// ImageSize ist die erwartete Bildgroesse in Pixeln
	ImageSize int `json:"image_size,omitempty"`

	// Author ist der HuggingFace Autor/Organisation
	Author string `json:"author,omitempty"`

	// Downloads ist die Anzahl Downloads (wenn bekannt)
	Downloads int64 `json:"downloads,omitempty"`
}

// HFModelsListResponse - Liste bekannter HuggingFace Modelle.
type HFModelsListResponse struct {
	// Models ist die Liste der bekannten Modelle
	Models []HFModelInfo `json:"models"`

	// Count ist die Anzahl der Modelle
	Count int `json:"count"`
}

// ============================================================================
// Cache Status Types
// ============================================================================

// CachedModel - Informationen zu einem gecachten Modell.
type CachedModel struct {
	// ModelID ist die HuggingFace Model-ID
	ModelID string `json:"model_id"`

	// Path ist der lokale Pfad zur Datei
	Path string `json:"path"`

	// SizeBytes ist die Groesse in Bytes
	SizeBytes int64 `json:"size_bytes"`

	// EncoderType ist der Encoder-Typ
	EncoderType string `json:"encoder_type"`

	// Revision ist die geladene Version/Revision
	Revision string `json:"revision,omitempty"`

	// CachedAt ist der Zeitstempel der Cache-Erstellung (Unix Timestamp)
	CachedAt int64 `json:"cached_at"`

	// LastAccessed ist der Zeitstempel des letzten Zugriffs (Unix Timestamp)
	LastAccessed int64 `json:"last_accessed,omitempty"`
}

// CacheStatus - Status des HuggingFace Modell-Caches.
// Endpoint: GET /api/vision/cache
type CacheStatus struct {
	// CacheDir ist das Cache-Verzeichnis
	CacheDir string `json:"cache_dir"`

	// TotalSize ist die Gesamtgroesse in Bytes
	TotalSize int64 `json:"total_size"`

	// ModelCount ist die Anzahl gecachter Modelle
	ModelCount int `json:"model_count"`

	// Models ist die Liste der gecachten Modelle
	Models []CachedModel `json:"models"`

	// MaxSize ist die maximale Cache-Groesse in Bytes
	MaxSize int64 `json:"max_size,omitempty"`

	// UsagePercent ist der Cache-Auslastungsprozentsatz
	UsagePercent float64 `json:"usage_percent,omitempty"`
}

// ============================================================================
// Cache Management Types
// ============================================================================

// ClearCacheRequest - Anfrage zum Leeren des Caches.
// Endpoint: DELETE /api/vision/cache
type ClearCacheRequest struct {
	// ModelID loescht nur ein bestimmtes Modell (optional)
	ModelID string `json:"model_id,omitempty"`

	// OlderThanDays loescht nur Modelle aelter als X Tage (optional)
	OlderThanDays int `json:"older_than_days,omitempty"`

	// DryRun simuliert das Loeschen ohne tatsaechliches Loeschen
	DryRun bool `json:"dry_run,omitempty"`
}

// ClearCacheResponse - Antwort nach dem Leeren des Caches.
type ClearCacheResponse struct {
	// Success gibt an, ob die Operation erfolgreich war
	Success bool `json:"success"`

	// DeletedCount ist die Anzahl geloeschter Modelle
	DeletedCount int `json:"deleted_count"`

	// FreedBytes ist die freigegebene Groesse in Bytes
	FreedBytes int64 `json:"freed_bytes"`

	// RemainingModels ist die Anzahl verbleibender Modelle
	RemainingModels int `json:"remaining_models"`

	// Message ist eine optionale Status-Nachricht
	Message string `json:"message,omitempty"`
}

// ============================================================================
// HuggingFace API Error Types
// ============================================================================

// HFAPIError - Strukturierter Fehler fuer HuggingFace API.
type HFAPIError struct {
	// Code ist der Fehler-Code
	Code string `json:"code"`

	// Message ist die Fehlermeldung
	Message string `json:"message"`

	// ModelID ist die betroffene Model-ID (falls zutreffend)
	ModelID string `json:"model_id,omitempty"`
}

// Error implementiert das error Interface.
func (e HFAPIError) Error() string {
	return e.Message
}

// HF-spezifische Fehler-Codes
const (
	// HFErrorModelNotFound - Modell nicht auf HuggingFace gefunden
	HFErrorModelNotFound = "HF_MODEL_NOT_FOUND"

	// HFErrorUnsupportedModel - Modell-Typ nicht unterstuetzt
	HFErrorUnsupportedModel = "HF_UNSUPPORTED_MODEL"

	// HFErrorDownloadFailed - Download fehlgeschlagen
	HFErrorDownloadFailed = "HF_DOWNLOAD_FAILED"

	// HFErrorConversionFailed - Konvertierung zu GGUF fehlgeschlagen
	HFErrorConversionFailed = "HF_CONVERSION_FAILED"

	// HFErrorCacheError - Cache-Fehler
	HFErrorCacheError = "HF_CACHE_ERROR"

	// HFErrorInvalidRequest - Ungueltiger Request
	HFErrorInvalidRequest = "HF_INVALID_REQUEST"

	// HFErrorRateLimited - Rate-Limit erreicht
	HFErrorRateLimited = "HF_RATE_LIMITED"

	// HFErrorAuthRequired - Authentifizierung erforderlich
	HFErrorAuthRequired = "HF_AUTH_REQUIRED"
)
