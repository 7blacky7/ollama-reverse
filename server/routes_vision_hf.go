// MODUL: routes_vision_hf
// ZWECK: HTTP Route-Definitionen und Handler-Struktur fuer HuggingFace Vision Model Endpoints
// INPUT: HTTP Requests mit HuggingFace Model-IDs
// OUTPUT: JSON Responses mit Modell-Status und Cache-Informationen
// NEBENEFFEKTE: Initialisiert Cache-Verzeichnis, scannt existierende Eintraege
// ABHAENGIGKEITEN: types_vision_hf (intern), handlers_vision_hf (intern), os, path/filepath, sync (stdlib)
// HINWEISE: Handler-Logik ist in handlers_vision_hf.go ausgelagert

package server

import (
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

// ============================================================================
// HuggingFace Handler Struktur
// ============================================================================

// HFVisionHandler verwaltet HuggingFace Vision Model Endpoints.
type HFVisionHandler struct {
	// cacheDir ist das Verzeichnis fuer gecachte Modelle
	cacheDir string

	// knownModels ist die Liste bekannter/unterstuetzter Modelle
	knownModels []HFModelInfo

	// cachedModels ist eine Map von Model-ID zu Cache-Info
	cachedModels map[string]*CachedModel

	// mu schuetzt gleichzeitige Zugriffe
	mu sync.RWMutex
}

// NewHFVisionHandler erstellt einen neuen HuggingFace Vision Handler.
// cacheDir ist das Verzeichnis fuer gecachte Modelle.
func NewHFVisionHandler(cacheDir string) *HFVisionHandler {
	if cacheDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			homeDir = "."
		}
		cacheDir = filepath.Join(homeDir, DefaultCacheDir, "vision-models")
	}

	handler := &HFVisionHandler{
		cacheDir:     cacheDir,
		cachedModels: make(map[string]*CachedModel),
		knownModels:  initKnownModels(),
	}

	// Cache-Verzeichnis erstellen falls noetig
	_ = os.MkdirAll(cacheDir, 0755)

	// Existierende Cache-Eintraege laden
	handler.scanCacheDir()

	return handler
}

// initKnownModels initialisiert die Liste bekannter HuggingFace Modelle.
func initKnownModels() []HFModelInfo {
	return []HFModelInfo{
		{
			ModelID:      "google/siglip-base-patch16-224",
			Type:         HFEncoderTypeSigLIP,
			Description:  "SigLIP Base ViT-B/16 mit 224x224 Bildgroesse",
			Supported:    true,
			EmbeddingDim: 768,
			ImageSize:    224,
			Author:       "google",
		},
		{
			ModelID:      "google/siglip-so400m-patch14-384",
			Type:         HFEncoderTypeSigLIP,
			Description:  "SigLIP SO400M mit 384x384 Bildgroesse, groesseres Modell",
			Supported:    true,
			EmbeddingDim: 1152,
			ImageSize:    384,
			Author:       "google",
		},
		{
			ModelID:      "openai/clip-vit-base-patch32",
			Type:         HFEncoderTypeCLIP,
			Description:  "CLIP ViT-B/32, schnelles Basis-Modell",
			Supported:    true,
			EmbeddingDim: 512,
			ImageSize:    224,
			Author:       "openai",
		},
		{
			ModelID:      "openai/clip-vit-large-patch14",
			Type:         HFEncoderTypeCLIP,
			Description:  "CLIP ViT-L/14, groesseres Modell mit besserer Qualitaet",
			Supported:    true,
			EmbeddingDim: 768,
			ImageSize:    224,
			Author:       "openai",
		},
		{
			ModelID:      "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
			Type:         HFEncoderTypeOpenCLIP,
			Description:  "OpenCLIP ViT-H/14, trainiert auf LAION-2B",
			Supported:    true,
			EmbeddingDim: 1024,
			ImageSize:    224,
			Author:       "laion",
		},
		{
			ModelID:      "nomic-ai/nomic-embed-vision-v1.5",
			Type:         HFEncoderTypeNomic,
			Description:  "Nomic Vision Encoder v1.5, multi-modal embedding",
			Supported:    true,
			EmbeddingDim: 768,
			ImageSize:    384,
			Author:       "nomic-ai",
		},
	}
}

// ============================================================================
// Route Registration
// ============================================================================

// RegisterHFRoutes registriert alle HuggingFace Vision Routes.
// Verwendet Go 1.22+ Routing-Syntax mit HTTP-Methoden-Praefixen.
func RegisterHFRoutes(mux *http.ServeMux, handler *HFVisionHandler) {
	// Model Loading
	mux.HandleFunc("POST /api/vision/load/hf", handler.handleLoadHFModel)

	// Model Listing
	mux.HandleFunc("GET /api/vision/models/hf", handler.handleListHFModels)

	// Cache Management
	mux.HandleFunc("GET /api/vision/cache", handler.handleCacheStatus)
	mux.HandleFunc("DELETE /api/vision/cache", handler.handleClearCache)
}

// RegisterHFRoutesLegacy registriert Routes ohne HTTP-Methoden-Praefixe.
// Fuer Kompatibilitaet mit aelteren Go-Versionen.
func RegisterHFRoutesLegacy(mux *http.ServeMux, handler *HFVisionHandler) {
	mux.HandleFunc("/api/vision/load/hf", handler.handleLoadHFModel)
	mux.HandleFunc("/api/vision/models/hf", handler.handleListHFModels)
	mux.HandleFunc("/api/vision/cache", handler.handleCacheStatus)
}
