// MODUL: helpers_vision_hf
// ZWECK: Hilfsfunktionen fuer HuggingFace Vision Handler (Cache, Pfade, JSON)
// INPUT: Model-IDs, Revisions, Cache-Verzeichnisse
// OUTPUT: Pfade, Cache-Eintraege, JSON-Responses
// NEBENEFFEKTE: Scannt Dateisystem, aktualisiert Cache-Timestamps
// ABHAENGIGKEITEN: types_vision_hf (intern), encoding/json, os, path/filepath, strings, time (stdlib)
// HINWEISE: Thread-sicher durch RWMutex auf HFVisionHandler

package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ============================================================================
// Modell-Suche
// ============================================================================

// findKnownModel sucht ein Modell in der Liste bekannter Modelle.
func (h *HFVisionHandler) findKnownModel(modelID string) *HFModelInfo {
	for i, model := range h.knownModels {
		if model.ModelID == modelID {
			return &h.knownModels[i]
		}
	}
	return nil
}

// ============================================================================
// Pfad-Generierung
// ============================================================================

// getModelPath generiert den Cache-Pfad fuer ein Modell.
func (h *HFVisionHandler) getModelPath(modelID, revision string) string {
	safeName := strings.ReplaceAll(modelID, "/", "_")
	safeName = strings.ReplaceAll(safeName, "\\", "_")

	if revision != "" {
		safeName = fmt.Sprintf("%s_%s", safeName, revision)
	}

	return filepath.Join(h.cacheDir, safeName+".gguf")
}

// ============================================================================
// Cache-Verwaltung
// ============================================================================

// updateLastAccessed aktualisiert den letzten Zugriffszeitpunkt.
func (h *HFVisionHandler) updateLastAccessed(modelID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if cached, ok := h.cachedModels[modelID]; ok {
		cached.LastAccessed = time.Now().Unix()
	}
}

// scanCacheDir scannt das Cache-Verzeichnis nach existierenden Modellen.
func (h *HFVisionHandler) scanCacheDir() {
	entries, err := os.ReadDir(h.cacheDir)
	if err != nil {
		return
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if !strings.HasSuffix(entry.Name(), ".gguf") {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		modelID := strings.TrimSuffix(entry.Name(), ".gguf")
		modelID = strings.ReplaceAll(modelID, "_", "/")

		var encoderType string
		if known := h.findKnownModel(modelID); known != nil {
			encoderType = known.Type
		}

		h.cachedModels[modelID] = &CachedModel{
			ModelID:      modelID,
			Path:         filepath.Join(h.cacheDir, entry.Name()),
			SizeBytes:    info.Size(),
			EncoderType:  encoderType,
			CachedAt:     info.ModTime().Unix(),
			LastAccessed: info.ModTime().Unix(),
		}
	}
}

// ============================================================================
// JSON Response Helper
// ============================================================================

// writeJSON schreibt eine JSON-Response.
func (h *HFVisionHandler) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeHFError schreibt eine Fehler-Response.
func (h *HFVisionHandler) writeHFError(w http.ResponseWriter, status int, code, message string) {
	h.writeJSON(w, status, HFAPIError{
		Code:    code,
		Message: message,
	})
}
