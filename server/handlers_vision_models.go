// MODUL: handlers_vision_models
// ZWECK: HTTP Handler Implementierungen fuer Vision Modell-Verwaltung (list, load, unload, info)
// INPUT: HTTP Requests (GET/POST) mit Modell-Namen und Pfaden
// OUTPUT: JSON Responses mit Modell-Listen und Status
// NEBENEFFEKTE: Laedt/Entlaedt Modelle, veraendert Handler-State
// ABHAENGIGKEITEN: routes_vision_models (intern), vision (intern), os (stdlib)
// HINWEISE: Thread-sicher, Modelle werden im VisionHandler gecacht

package server

import (
	"fmt"
	"net/http"
	"os"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// GET /api/vision/models - Verfuegbare Modell-Typen auflisten
// ============================================================================

// HandleListModels verarbeitet GET /api/vision/models
func (h *VisionHandler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

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

	if err := h.validateLoadRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

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
	if _, err := os.Stat(req.Path); os.IsNotExist(err) {
		return nil, "", fmt.Errorf("model file not found: %s", req.Path)
	}

	modelType := req.Type
	if modelType == "" {
		detected, err := vision.AutoDetectEncoder(req.Path)
		if err != nil {
			return nil, "", fmt.Errorf("auto-detection failed: %v", err)
		}
		modelType = detected
	}

	if !h.registry.Has(modelType) {
		return nil, "", fmt.Errorf("unknown model type: %s", modelType)
	}

	encoder, err := h.registry.CreateWithDefaults(modelType, req.Path)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create encoder: %v", err)
	}

	h.mu.Lock()
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

	if req.Model == "" {
		h.writeVisionError(w, http.StatusBadRequest, "model name is required", "VALIDATION_ERROR")
		return
	}

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
