//go:build vision
// MODUL: handlers_vision_hf
// ZWECK: HTTP Handler Implementierungen fuer HuggingFace Vision Model Endpoints
// INPUT: HTTP Requests (POST/GET/DELETE) mit Model-IDs und Cache-Parametern
// OUTPUT: JSON Responses mit Modell-Status, Listen und Cache-Informationen
// NEBENEFFEKTE: Laedt/Cacht HF-Modelle, modifiziert Cache-Eintraege
// ABHAENGIGKEITEN: types_vision_hf (intern), huggingface (intern), encoding/json, net/http, time (stdlib)
// HINWEISE: Cache-Hilfsfunktionen sind in helpers_vision_hf.go ausgelagert

package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/ollama/ollama/huggingface"
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

	// Tatsaechlicher Download von HuggingFace
	slog.Info("Starte HuggingFace Download", "model_id", req.ModelID, "revision", req.Revision)

	revision := req.Revision
	if revision == "" {
		revision = "main"
	}

	// Download mit Progress-Logging
	downloadOpts := []huggingface.DownloadOption{
		huggingface.WithDownloadRevision(revision),
		huggingface.WithDownloadProgress(func(downloaded, total int64) {
			// Division durch Null vermeiden (total=0 bei gecachten Dateien)
			if total > 0 {
				percent := float64(downloaded) / float64(total) * 100
				slog.Info("Download-Fortschritt", "model_id", req.ModelID, "percent", fmt.Sprintf("%.1f%%", percent))
			} else if downloaded > 0 {
				slog.Info("Download-Fortschritt", "model_id", req.ModelID, "bytes", downloaded)
			}
		}),
	}

	result, err := huggingface.DownloadModel(req.ModelID, downloadOpts...)
	if err != nil {
		slog.Error("HuggingFace Download fehlgeschlagen", "model_id", req.ModelID, "error", err)
		h.writeHFError(w, http.StatusInternalServerError, HFErrorDownloadFailed,
			fmt.Sprintf("Download fehlgeschlagen: %v", err))
		return
	}

	slog.Info("HuggingFace Download erfolgreich",
		"model_id", req.ModelID,
		"cache_path", result.CachePath,
		"total_size", result.TotalSize,
		"download_time", result.DownloadTime,
		"files_count", len(result.Files))

	// Cache-Eintrag mit echten Daten aktualisieren
	h.mu.Lock()
	h.cachedModels[req.ModelID] = &CachedModel{
		ModelID:      req.ModelID,
		Path:         result.CachePath,
		SizeBytes:    result.TotalSize,
		EncoderType:  modelInfo.Type,
		Revision:     revision,
		CachedAt:     time.Now().Unix(),
		LastAccessed: time.Now().Unix(),
	}
	h.mu.Unlock()

	response := LoadHFModelResponse{
		Status:       HFModelStatusLoaded,
		EncoderType:  modelInfo.Type,
		ModelPath:    result.CachePath,
		CacheHit:     false,
		ModelID:      req.ModelID,
		EmbeddingDim: modelInfo.EmbeddingDim,
		Message:      fmt.Sprintf("Modell erfolgreich geladen (%d Dateien, %.2f MB)", len(result.Files), float64(result.TotalSize)/1024/1024),
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
