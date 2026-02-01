// Package server implementiert die REST API Handler fuer SigLIP.
// Diese Datei enthaelt den SigLIPHandler und ModelManager.
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/siglip"
)

// ============================================================================
// Model Manager - Cached Model Loading mit Thread-Safety
// ============================================================================

// ModelManager verwaltet geladene SigLIP-Modelle mit Caching.
type ModelManager struct {
	mu       sync.RWMutex
	models   map[string]*cachedModel
	modelDir string
}

// cachedModel repraesentiert ein gecachtes Modell mit Metadaten.
type cachedModel struct {
	model    *siglip.Model
	name     string
	path     string
	size     int64
	useCount int64
}

// NewModelManager erstellt einen neuen ModelManager.
func NewModelManager(modelDir string) *ModelManager {
	return &ModelManager{
		models:   make(map[string]*cachedModel),
		modelDir: modelDir,
	}
}

// GetModel holt ein Modell aus dem Cache oder laedt es neu.
func (mm *ModelManager) GetModel(name string) (*siglip.Model, error) {
	// Zuerst mit Read-Lock pruefen
	mm.mu.RLock()
	if cached, ok := mm.models[name]; ok {
		cached.useCount++
		mm.mu.RUnlock()
		return cached.model, nil
	}
	mm.mu.RUnlock()

	// Modell laden (Write-Lock)
	mm.mu.Lock()
	defer mm.mu.Unlock()

	// Double-Check nach Write-Lock
	if cached, ok := mm.models[name]; ok {
		cached.useCount++
		return cached.model, nil
	}

	// Modell-Pfad bestimmen
	modelPath := mm.resolveModelPath(name)
	if modelPath == "" {
		return nil, fmt.Errorf("model not found: %s", name)
	}

	// Datei-Info holen
	fileInfo, err := os.Stat(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat model file: %w", err)
	}

	// Modell laden
	model, err := siglip.LoadModel(modelPath,
		siglip.WithBackend(siglip.BackendCPU),
		siglip.WithLogLevel(siglip.LogWarn),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Im Cache speichern
	mm.models[name] = &cachedModel{
		model:    model,
		name:     name,
		path:     modelPath,
		size:     fileInfo.Size(),
		useCount: 1,
	}

	return model, nil
}

// ListModels gibt eine Liste aller verfuegbaren Modelle zurueck.
func (mm *ModelManager) ListModels() []api.ModelInfo {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	var models []api.ModelInfo

	// Geladene Modelle aus Cache
	for name, cached := range mm.models {
		models = append(models, api.ModelInfo{
			Name:         name,
			Size:         cached.size,
			EmbeddingDim: cached.model.EmbeddingDim(),
			ImageSize:    cached.model.ImageSize(),
			Type:         cached.model.ModelType().String(),
			Backend:      "CPU", // TODO: Backend vom Modell holen
		})
	}

	// GGUF-Dateien im Model-Verzeichnis scannen
	if mm.modelDir != "" {
		entries, err := os.ReadDir(mm.modelDir)
		if err == nil {
			for _, entry := range entries {
				if entry.IsDir() {
					continue
				}
				if !strings.HasSuffix(entry.Name(), ".gguf") {
					continue
				}

				name := strings.TrimSuffix(entry.Name(), ".gguf")

				// Pruefen ob bereits in der Liste
				found := false
				for _, m := range models {
					if m.Name == name {
						found = true
						break
					}
				}
				if found {
					continue
				}

				info, err := entry.Info()
				if err != nil {
					continue
				}

				models = append(models, api.ModelInfo{
					Name: name,
					Size: info.Size(),
				})
			}
		}
	}

	return models
}

// UnloadModel entlaedt ein Modell aus dem Cache.
func (mm *ModelManager) UnloadModel(name string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	cached, ok := mm.models[name]
	if !ok {
		return fmt.Errorf("model not loaded: %s", name)
	}

	if err := cached.model.Close(); err != nil {
		return fmt.Errorf("failed to close model: %w", err)
	}

	delete(mm.models, name)
	return nil
}

// Close schliesst alle geladenen Modelle.
func (mm *ModelManager) Close() error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	var firstErr error
	for name, cached := range mm.models {
		if err := cached.model.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(mm.models, name)
	}

	return firstErr
}

// resolveModelPath findet den Pfad zum Modell.
func (mm *ModelManager) resolveModelPath(name string) string {
	// Direkter Pfad
	if _, err := os.Stat(name); err == nil {
		return name
	}

	// Mit .gguf Extension
	if !strings.HasSuffix(name, ".gguf") {
		withExt := name + ".gguf"
		if _, err := os.Stat(withExt); err == nil {
			return withExt
		}
	}

	// Im Model-Verzeichnis
	if mm.modelDir != "" {
		modelPath := filepath.Join(mm.modelDir, name)
		if _, err := os.Stat(modelPath); err == nil {
			return modelPath
		}

		if !strings.HasSuffix(name, ".gguf") {
			modelPath = filepath.Join(mm.modelDir, name+".gguf")
			if _, err := os.Stat(modelPath); err == nil {
				return modelPath
			}
		}
	}

	return ""
}

// ============================================================================
// SigLIP Handler
// ============================================================================

// SigLIPHandler verwaltet die SigLIP REST API Endpoints.
type SigLIPHandler struct {
	manager *ModelManager
}

// NewSigLIPHandler erstellt einen neuen SigLIPHandler.
func NewSigLIPHandler(modelDir string) *SigLIPHandler {
	return &SigLIPHandler{
		manager: NewModelManager(modelDir),
	}
}

// Close schliesst alle Ressourcen.
func (h *SigLIPHandler) Close() error {
	return h.manager.Close()
}

// ============================================================================
// Helper Functions
// ============================================================================

// writeJSON schreibt eine JSON-Response.
func (h *SigLIPHandler) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeError schreibt eine Fehler-Response.
func (h *SigLIPHandler) writeError(w http.ResponseWriter, status int, message, code string) {
	h.writeJSON(w, status, api.ErrorResponse{
		Error: message,
		Code:  code,
	})
}

// decodeJSON dekodiert JSON aus dem Request-Body.
func decodeJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

// ============================================================================
// Server Convenience Function
// ============================================================================

// StartSigLIPServer startet einen HTTP-Server fuer SigLIP.
func StartSigLIPServer(addr, modelDir string) error {
	handler := NewSigLIPHandler(modelDir)
	defer handler.Close()

	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	// Health-Check Endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	return http.ListenAndServe(addr, mux)
}
