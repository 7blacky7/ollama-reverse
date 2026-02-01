// MODUL: routes_vision_models
// ZWECK: REST API Handler fuer Vision Modell-Verwaltung (list, load, unload)
// INPUT: HTTP Requests mit Modell-Namen und Pfaden
// OUTPUT: JSON Responses mit Modell-Listen und Status
// NEBENEFFEKTE: Laedt/Entlaedt Modelle, veraendert Handler-State
// ABHAENGIGKEITEN: vision (intern), encoding/json, net/http (stdlib)
// HINWEISE: Thread-sicher, Modelle werden im VisionHandler gecacht

package server

import (
	"fmt"
	"net/http"
	"os"

	"github.com/ollama/ollama/vision"
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

// ============================================================================
// GET /api/vision/models - Verfuegbare Modell-Typen auflisten
// ============================================================================

// HandleListModels verarbeitet GET /api/vision/models
func (h *VisionHandler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	// Registrierte Modell-Typen aus der Registry holen
	modelTypes := h.registry.List()

	h.writeVisionJSON(w, http.StatusOK, VisionModelsResponse{
		Models: modelTypes,
	})
}

// ============================================================================
// GET /api/vision/models/loaded - Geladene Modelle auflisten
// ============================================================================

// HandleListLoadedModels verarbeitet GET /api/vision/models/loaded
func (h *VisionHandler) HandleListLoadedModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	models := h.getLoadedModelsInfo()

	h.writeVisionJSON(w, http.StatusOK, VisionLoadedModelsResponse{
		Models: models,
		Count:  len(models),
	})
}

// getLoadedModelsInfo sammelt Informationen ueber geladene Modelle
func (h *VisionHandler) getLoadedModelsInfo() []VisionModelInfo {
	h.mu.RLock()
	defer h.mu.RUnlock()

	models := make([]VisionModelInfo, 0, len(h.models))
	for name, encoder := range h.models {
		info := encoder.ModelInfo()
		models = append(models, VisionModelInfo{
			Name:         name,
			Type:         info.Type,
			Path:         h.paths[name],
			EmbeddingDim: info.EmbeddingDim,
			ImageSize:    info.ImageSize,
		})
	}
	return models
}

// ============================================================================
// POST /api/vision/load - Modell laden
// ============================================================================

// HandleLoadModel verarbeitet POST /api/vision/load
func (h *VisionHandler) HandleLoadModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionLoadRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	// Validierung
	if err := h.validateLoadRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Modell laden
	encoder, modelType, err := h.loadModel(req)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "LOAD_ERROR")
		return
	}

	info := encoder.ModelInfo()
	h.writeVisionJSON(w, http.StatusOK, VisionLoadResponse{
		Model:        req.Model,
		Type:         modelType,
		EmbeddingDim: info.EmbeddingDim,
		Success:      true,
	})
}

// validateLoadRequest validiert einen Load-Request
func (h *VisionHandler) validateLoadRequest(req VisionLoadRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model name is required")
	}
	if req.Path == "" {
		return fmt.Errorf("model path is required")
	}
	return nil
}

// loadModel laedt ein Modell und speichert es im Cache
func (h *VisionHandler) loadModel(req VisionLoadRequest) (vision.VisionEncoder, string, error) {
	// Pruefen ob Datei existiert
	if _, err := os.Stat(req.Path); os.IsNotExist(err) {
		return nil, "", fmt.Errorf("model file not found: %s", req.Path)
	}

	// Modell-Typ bestimmen (auto-detect oder aus Request)
	modelType := req.Type
	if modelType == "" {
		detected, err := vision.AutoDetectEncoder(req.Path)
		if err != nil {
			return nil, "", fmt.Errorf("auto-detection failed: %v", err)
		}
		modelType = detected
	}

	// Pruefen ob Typ registriert ist
	if !h.registry.Has(modelType) {
		return nil, "", fmt.Errorf("unknown model type: %s", modelType)
	}

	// Encoder erstellen
	encoder, err := h.registry.CreateWithDefaults(modelType, req.Path)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create encoder: %v", err)
	}

	// Im Cache speichern
	h.mu.Lock()
	// Altes Modell entladen falls vorhanden
	if old, exists := h.models[req.Model]; exists {
		old.Close()
	}
	h.models[req.Model] = encoder
	h.paths[req.Model] = req.Path
	h.mu.Unlock()

	return encoder, modelType, nil
}

// ============================================================================
// POST /api/vision/unload - Modell entladen
// ============================================================================

// HandleUnloadModel verarbeitet POST /api/vision/unload
func (h *VisionHandler) HandleUnloadModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionUnloadRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	// Validierung
	if req.Model == "" {
		h.writeVisionError(w, http.StatusBadRequest, "model name is required", "VALIDATION_ERROR")
		return
	}

	// Modell entladen
	if err := h.unloadModel(req.Model); err != nil {
		h.writeVisionError(w, http.StatusNotFound, err.Error(), "UNLOAD_ERROR")
		return
	}

	h.writeVisionJSON(w, http.StatusOK, VisionUnloadResponse{
		Model:   req.Model,
		Success: true,
	})
}

// unloadModel entlaedt ein Modell aus dem Cache
func (h *VisionHandler) unloadModel(name string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	encoder, exists := h.models[name]
	if !exists {
		return fmt.Errorf("model not loaded: %s", name)
	}

	if err := encoder.Close(); err != nil {
		return fmt.Errorf("failed to close model: %v", err)
	}

	delete(h.models, name)
	delete(h.paths, name)
	return nil
}

// ============================================================================
// GET /api/vision/info - Vision System Info
// ============================================================================

// VisionInfoResponse - Response fuer GET /api/vision/info
type VisionInfoResponse struct {
	RegisteredTypes []string `json:"registered_types"` // Registrierte Encoder-Typen
	LoadedCount     int      `json:"loaded_count"`     // Anzahl geladener Modelle
	Version         string   `json:"version"`          // Vision API Version
}

// HandleInfo verarbeitet GET /api/vision/info
func (h *VisionHandler) HandleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	h.mu.RLock()
	loadedCount := len(h.models)
	h.mu.RUnlock()

	h.writeVisionJSON(w, http.StatusOK, VisionInfoResponse{
		RegisteredTypes: h.registry.List(),
		LoadedCount:     loadedCount,
		Version:         "1.0.0",
	})
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
