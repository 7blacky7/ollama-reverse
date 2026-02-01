// Package server implementiert die REST API Handler fuer SigLIP.
package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
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
// HTTP Handler Functions
// ============================================================================

// HandleEmbedImage verarbeitet POST /api/embed/image
func (h *SigLIPHandler) HandleEmbedImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req api.EmbedImageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	// Validierung
	if req.Model == "" {
		h.writeError(w, http.StatusBadRequest, "model is required", "MISSING_MODEL")
		return
	}
	if req.Image == "" {
		h.writeError(w, http.StatusBadRequest, "image is required", "MISSING_IMAGE")
		return
	}

	// Base64 dekodieren
	imageData, err := base64.StdEncoding.DecodeString(req.Image)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid base64 image: %v", err), "INVALID_IMAGE")
		return
	}

	// Modell laden
	model, err := h.manager.GetModel(req.Model)
	if err != nil {
		h.writeError(w, http.StatusNotFound, fmt.Sprintf("model error: %v", err), "MODEL_ERROR")
		return
	}

	// Embedding generieren
	embedding, err := model.Encode(imageData)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("encoding failed: %v", err), "ENCODING_ERROR")
		return
	}

	// Response
	resp := api.EmbedImageResponse{
		Embedding: embedding.ToFloat32(),
		Model:     req.Model,
		Dimension: embedding.Size(),
	}

	h.writeJSON(w, http.StatusOK, resp)
}

// HandleEmbedBatch verarbeitet POST /api/embed/batch
func (h *SigLIPHandler) HandleEmbedBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req api.EmbedBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	// Validierung
	if req.Model == "" {
		h.writeError(w, http.StatusBadRequest, "model is required", "MISSING_MODEL")
		return
	}
	if len(req.Images) == 0 {
		h.writeError(w, http.StatusBadRequest, "images are required", "MISSING_IMAGES")
		return
	}

	// Base64 dekodieren
	imagesData := make([][]byte, len(req.Images))
	for i, imgBase64 := range req.Images {
		data, err := base64.StdEncoding.DecodeString(imgBase64)
		if err != nil {
			h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid base64 image at index %d: %v", i, err), "INVALID_IMAGE")
			return
		}
		imagesData[i] = data
	}

	// Modell laden
	model, err := h.manager.GetModel(req.Model)
	if err != nil {
		h.writeError(w, http.StatusNotFound, fmt.Sprintf("model error: %v", err), "MODEL_ERROR")
		return
	}

	// Batch-Encoding
	embeddings, err := model.EncodeBatch(imagesData)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("encoding failed: %v", err), "ENCODING_ERROR")
		return
	}

	// Response erstellen
	embeddingsFloat := make([][]float32, len(embeddings))
	var dimension int
	for i, emb := range embeddings {
		if emb != nil {
			embeddingsFloat[i] = emb.ToFloat32()
			if dimension == 0 {
				dimension = emb.Size()
			}
		}
	}

	resp := api.EmbedBatchResponse{
		Embeddings: embeddingsFloat,
		Model:      req.Model,
		Count:      len(embeddings),
		Dimension:  dimension,
	}

	h.writeJSON(w, http.StatusOK, resp)
}

// HandleSimilarity verarbeitet POST /api/similarity
func (h *SigLIPHandler) HandleSimilarity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req api.SimilarityRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	// Validierung
	if req.Model == "" {
		h.writeError(w, http.StatusBadRequest, "model is required", "MISSING_MODEL")
		return
	}
	if req.Image == "" {
		h.writeError(w, http.StatusBadRequest, "image is required", "MISSING_IMAGE")
		return
	}
	if len(req.Candidates) == 0 {
		h.writeError(w, http.StatusBadRequest, "candidates are required", "MISSING_CANDIDATES")
		return
	}

	// Query-Image dekodieren
	queryData, err := base64.StdEncoding.DecodeString(req.Image)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid base64 query image: %v", err), "INVALID_IMAGE")
		return
	}

	// Kandidaten dekodieren
	candidatesData := make([][]byte, len(req.Candidates))
	for i, candBase64 := range req.Candidates {
		data, err := base64.StdEncoding.DecodeString(candBase64)
		if err != nil {
			h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid base64 candidate at index %d: %v", i, err), "INVALID_IMAGE")
			return
		}
		candidatesData[i] = data
	}

	// Modell laden
	model, err := h.manager.GetModel(req.Model)
	if err != nil {
		h.writeError(w, http.StatusNotFound, fmt.Sprintf("model error: %v", err), "MODEL_ERROR")
		return
	}

	// Query-Embedding generieren
	queryEmb, err := model.Encode(queryData)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("query encoding failed: %v", err), "ENCODING_ERROR")
		return
	}

	// Kandidaten-Embeddings generieren
	candidateEmbs, err := model.EncodeBatch(candidatesData)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("candidates encoding failed: %v", err), "ENCODING_ERROR")
		return
	}

	// Similarity berechnen
	results := make([]api.SimilarityResult, len(candidateEmbs))
	for i, candEmb := range candidateEmbs {
		if candEmb == nil {
			results[i] = api.SimilarityResult{Index: i, Score: 0}
			continue
		}
		results[i] = api.SimilarityResult{
			Index: i,
			Score: queryEmb.CosineSimilarity(candEmb),
		}
	}

	// Nach Score sortieren (absteigend)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Top-K begrenzen
	topK := req.TopK
	if topK <= 0 || topK > len(results) {
		topK = len(results)
	}
	results = results[:topK]

	resp := api.SimilarityResponse{
		Results: results,
		Model:   req.Model,
		Count:   len(results),
	}

	h.writeJSON(w, http.StatusOK, resp)
}

// HandleListModels verarbeitet GET /api/siglip/models
func (h *SigLIPHandler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	models := h.manager.ListModels()

	resp := api.ListModelsResponse{
		Models: models,
	}

	h.writeJSON(w, http.StatusOK, resp)
}

// HandleSigLIPInfo verarbeitet GET /api/siglip/info
func (h *SigLIPHandler) HandleSigLIPInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	backends := siglip.AvailableBackends()
	backendNames := make([]string, len(backends))
	for i, b := range backends {
		backendNames[i] = b.String()
	}

	resp := api.SigLIPInfoResponse{
		Version:           siglip.Version(),
		BuildInfo:         siglip.BuildInfo(),
		SystemInfo:        siglip.SystemInfo(),
		AvailableBackends: backendNames,
	}

	h.writeJSON(w, http.StatusOK, resp)
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

// ============================================================================
// Router Setup
// ============================================================================

// RegisterRoutes registriert alle SigLIP-Routes auf einem http.ServeMux.
func (h *SigLIPHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/embed/image", h.HandleEmbedImage)
	mux.HandleFunc("/api/embed/batch", h.HandleEmbedBatch)
	mux.HandleFunc("/api/similarity", h.HandleSimilarity)
	mux.HandleFunc("/api/siglip/models", h.HandleListModels)
	mux.HandleFunc("/api/siglip/info", h.HandleSigLIPInfo)
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
