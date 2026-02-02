// MODUL: routes_vision_models
// ZWECK: REST API Typen und Route-Registrierung fuer Vision Modell-Verwaltung
// INPUT: HTTP Requests mit Modell-Namen und Pfaden
// OUTPUT: JSON Responses mit Modell-Listen und Status
// NEBENEFFEKTE: Keine (nur Definitionen)
// ABHAENGIGKEITEN: handlers_vision_models (intern), net/http (stdlib)
// HINWEISE: Handler-Implementierungen sind in handlers_vision_models.go ausgelagert

package server

import (
	"net/http"
)

// ============================================================================
// Request/Response Types fuer Modell-Verwaltung
// ============================================================================

// VisionModelsResponse - Response fuer GET /api/vision/models
type VisionModelsResponse struct {
	Models []string `json:"models"` // Liste verfuegbarer Modell-Typen
}

// VisionLoadedModelsResponse - Response fuer GET /api/vision/models/loaded
type VisionLoadedModelsResponse struct {
	Models []VisionModelInfo `json:"models"` // Liste geladener Modelle
	Count  int               `json:"count"`  // Anzahl geladener Modelle
}

// VisionModelInfo - Informationen zu einem geladenen Modell
type VisionModelInfo struct {
	Name         string `json:"name"`          // Modell-Name/Alias
	Type         string `json:"type"`          // Modell-Typ (clip, siglip, etc.)
	Path         string `json:"path"`          // Pfad zur Modell-Datei
	EmbeddingDim int    `json:"embedding_dim"` // Embedding-Dimension
	ImageSize    int    `json:"image_size"`    // Erwartete Bildgroesse
}

// VisionLoadRequest - Request fuer POST /api/vision/load
type VisionLoadRequest struct {
	Model string `json:"model"` // Modell-Name/Alias
	Path  string `json:"path"`  // Pfad zur GGUF-Datei
	Type  string `json:"type"`  // Modell-Typ (optional, wird auto-detected)
}

// VisionLoadResponse - Response fuer POST /api/vision/load
type VisionLoadResponse struct {
	Model        string `json:"model"`         // Geladenes Modell
	Type         string `json:"type"`          // Modell-Typ
	EmbeddingDim int    `json:"embedding_dim"` // Embedding-Dimension
	Success      bool   `json:"success"`       // Erfolg-Status
}

// VisionUnloadRequest - Request fuer POST /api/vision/unload
type VisionUnloadRequest struct {
	Model string `json:"model"` // Modell-Name zum Entladen
}

// VisionUnloadResponse - Response fuer POST /api/vision/unload
type VisionUnloadResponse struct {
	Model   string `json:"model"`   // Entladenes Modell
	Success bool   `json:"success"` // Erfolg-Status
}

// VisionInfoResponse - Response fuer GET /api/vision/info
type VisionInfoResponse struct {
	RegisteredTypes []string `json:"registered_types"` // Registrierte Encoder-Typen
	LoadedCount     int      `json:"loaded_count"`     // Anzahl geladener Modelle
	Version         string   `json:"version"`          // Vision API Version
}

// ============================================================================
// Router Registration fuer Modell-Verwaltung
// ============================================================================

// RegisterVisionModelRoutes registriert alle Modell-Verwaltungs-Routes
func (h *VisionHandler) RegisterVisionModelRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/vision/models", h.HandleListModels)
	mux.HandleFunc("/api/vision/models/loaded", h.HandleListLoadedModels)
	mux.HandleFunc("/api/vision/load", h.HandleLoadModel)
	mux.HandleFunc("/api/vision/unload", h.HandleUnloadModel)
	mux.HandleFunc("/api/vision/info", h.HandleInfo)
}

// RegisterAllVisionRoutes registriert alle Vision-Routes (Encoding + Modelle)
func (h *VisionHandler) RegisterAllVisionRoutes(mux *http.ServeMux) {
	h.RegisterVisionRoutes(mux)
	h.RegisterVisionModelRoutes(mux)
}
