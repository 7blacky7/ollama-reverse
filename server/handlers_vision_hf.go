//go:build vision
// MODUL: handlers_vision_hf
// ZWECK: HTTP Handler Implementierungen fuer HuggingFace Vision Model Endpoints
// INPUT: HTTP Requests (POST/GET/DELETE) mit Model-IDs und Cache-Parametern
// OUTPUT: JSON Responses mit Modell-Status, Listen und Cache-Informationen
// NEBENEFFEKTE: Laedt/Cacht HF-Modelle, modifiziert Cache-Eintraege
// ABHAENGIGKEITEN: types_vision_hf (intern), encoding/json, net/http, time (stdlib)
// HINWEISE: Cache-Hilfsfunktionen sind in helpers_vision_hf.go ausgelagert

package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

// ============================================================================
// HTTP Handler: Model Loading
// ============================================================================

// handleLoadHFModel verarbeitet POST /api/vision/load/hf.
// Laedt ein Vision-Modell von HuggingFace.
func (h *HFVisionHandler) handleLoadHFModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeHFError(w, http.StatusMethodNotAllowed, HFErrorInvalidRequest,
			"Methode nicht erlaubt, verwende POST")
		return
	}

	var req LoadHFModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeHFError(w, http.StatusBadRequest, HFErrorInvalidRequest,
			fmt.Sprintf("Ungueltiger Request: %v", err))
		return
	}

	if req.ModelID == "" {
		h.writeHFError(w, http.StatusBadRequest, HFErrorInvalidRequest,
			"model_id ist erforderlich")
		return
	}

	modelInfo := h.findKnownModel(req.ModelID)
	if modelInfo == nil {
		h.writeHFError(w, http.StatusBadRequest, HFErrorUnsupportedModel,
			fmt.Sprintf("Modell nicht unterstuetzt: %s", req.ModelID))
		return
	}

	h.mu.RLock()
	cached, cacheHit := h.cachedModels[req.ModelID]
	h.mu.RUnlock()

	if cacheHit && !req.Force {
		h.updateLastAccessed(req.ModelID)
		response := LoadHFModelResponse{
			Status:       HFModelStatusCached,
			EncoderType:  modelInfo.Type,
			ModelPath:    cached.Path,
			CacheHit:     true,
			ModelID:      req.ModelID,
			EmbeddingDim: modelInfo.EmbeddingDim,
			Message:      "Modell aus Cache geladen",
		}
		h.writeJSON(w, http.StatusOK, response)
		return
	}

	modelPath := h.getModelPath(req.ModelID, req.Revision)

	h.mu.Lock()
	h.cachedModels[req.ModelID] = &CachedModel{
		ModelID:      req.ModelID,
		Path:         modelPath,
		SizeBytes:    0,
		EncoderType:  modelInfo.Type,
		Revision:     req.Revision,
		CachedAt:     time.Now().Unix(),
		LastAccessed: time.Now().Unix(),
	}
	h.mu.Unlock()

	response := LoadHFModelResponse{
		Status:       HFModelStatusLoaded,
		EncoderType:  modelInfo.Type,
		ModelPath:    modelPath,
		CacheHit:     false,
		ModelID:      req.ModelID,
		EmbeddingDim: modelInfo.EmbeddingDim,
		Message:      "Modell erfolgreich geladen",
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// HTTP Handler: Model Listing
// ============================================================================

// handleListHFModels verarbeitet GET /api/vision/models/hf.
func (h *HFVisionHandler) handleListHFModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeHFError(w, http.StatusMethodNotAllowed, HFErrorInvalidRequest,
			"Methode nicht erlaubt, verwende GET")
		return
	}

	typeFilter := r.URL.Query().Get("type")
	supportedOnly := r.URL.Query().Get("supported") == "true"

	var filteredModels []HFModelInfo
	for _, model := range h.knownModels {
		if typeFilter != "" && model.Type != typeFilter {
			continue
		}
		if supportedOnly && !model.Supported {
			continue
		}
		filteredModels = append(filteredModels, model)
	}

	response := HFModelsListResponse{
		Models: filteredModels,
		Count:  len(filteredModels),
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// HTTP Handler: Cache Status
// ============================================================================

// handleCacheStatus verarbeitet GET /api/vision/cache.
func (h *HFVisionHandler) handleCacheStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeHFError(w, http.StatusMethodNotAllowed, HFErrorInvalidRequest,
			"Methode nicht erlaubt, verwende GET")
		return
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	var totalSize int64
	models := make([]CachedModel, 0, len(h.cachedModels))

	for _, cached := range h.cachedModels {
		totalSize += cached.SizeBytes
		models = append(models, *cached)
	}

	var usagePercent float64
	if MaxCacheSizeBytes > 0 {
		usagePercent = float64(totalSize) / float64(MaxCacheSizeBytes) * 100.0
	}

	response := CacheStatus{
		CacheDir:     h.cacheDir,
		TotalSize:    totalSize,
		ModelCount:   len(h.cachedModels),
		Models:       models,
		MaxSize:      MaxCacheSizeBytes,
		UsagePercent: usagePercent,
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// HTTP Handler: Cache Clear
// ============================================================================

// handleClearCache verarbeitet DELETE /api/vision/cache.
func (h *HFVisionHandler) handleClearCache(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		h.writeHFError(w, http.StatusMethodNotAllowed, HFErrorInvalidRequest,
			"Methode nicht erlaubt, verwende DELETE")
		return
	}

	var req ClearCacheRequest
	_ = json.NewDecoder(r.Body).Decode(&req)

	h.mu.Lock()
	defer h.mu.Unlock()

	var deletedCount int
	var freedBytes int64
	now := time.Now()

	toDelete := make([]string, 0)

	for modelID, cached := range h.cachedModels {
		if req.ModelID != "" && modelID != req.ModelID {
			continue
		}
		if req.OlderThanDays > 0 {
			cachedTime := time.Unix(cached.CachedAt, 0)
			ageDays := int(now.Sub(cachedTime).Hours() / 24)
			if ageDays < req.OlderThanDays {
				continue
			}
		}
		toDelete = append(toDelete, modelID)
		freedBytes += cached.SizeBytes
	}

	if req.DryRun {
		response := ClearCacheResponse{
			Success:         true,
			DeletedCount:    len(toDelete),
			FreedBytes:      freedBytes,
			RemainingModels: len(h.cachedModels) - len(toDelete),
			Message:         "Dry-Run: Keine Dateien wurden geloescht",
		}
		h.writeJSON(w, http.StatusOK, response)
		return
	}

	for _, modelID := range toDelete {
		cached := h.cachedModels[modelID]
		if cached.Path != "" {
			_ = os.Remove(cached.Path)
		}
		delete(h.cachedModels, modelID)
		deletedCount++
	}

	response := ClearCacheResponse{
		Success:         true,
		DeletedCount:    deletedCount,
		FreedBytes:      freedBytes,
		RemainingModels: len(h.cachedModels),
		Message:         fmt.Sprintf("%d Modell(e) aus dem Cache entfernt", deletedCount),
	}

	h.writeJSON(w, http.StatusOK, response)
}
