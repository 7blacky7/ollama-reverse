// Package server implementiert die REST API Routes fuer SigLIP.
// Diese Datei enthaelt die HTTP Handler Funktionen und Route-Registrierung.
package server

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"sort"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/siglip"
)

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
	if err := decodeJSON(r, &req); err != nil {
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
	if err := decodeJSON(r, &req); err != nil {
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
	if err := decodeJSON(r, &req); err != nil {
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
