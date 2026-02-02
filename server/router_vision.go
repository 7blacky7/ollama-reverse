//go:build vision
// MODUL: router_vision
// ZWECK: Registriert Vision API Endpoints und definiert VisionHandler
// INPUT: http.ServeMux, VisionHandler
// OUTPUT: Konfigurierter HTTP-Router
// NEBENEFFEKTE: Registriert HTTP-Routen
// ABHAENGIGKEITEN: handler_siglip (intern, ModelManager)
// HINWEISE: Nutzt bestehenden ModelManager fuer Modell-Verwaltung

package server

import (
	"net/http"
)

// ============================================================================
// Vision Handler
// ============================================================================

// VisionHandler verwaltet die Vision API Endpoints.
// Verwendet ModelManager fuer das Laden und Cachen von Modellen.
type VisionHandler struct {
	manager *ModelManager
}

// NewVisionHandler erstellt einen neuen VisionHandler.
// modelDir ist das Verzeichnis fuer Modell-Dateien.
func NewVisionHandler(modelDir string) *VisionHandler {
	return &VisionHandler{
		manager: NewModelManager(modelDir),
	}
}

// NewVisionHandlerWithManager erstellt einen VisionHandler mit bestehendem Manager.
// Ermoeglicht das Teilen des ModelManagers zwischen Handlern.
func NewVisionHandlerWithManager(manager *ModelManager) *VisionHandler {
	return &VisionHandler{
		manager: manager,
	}
}

// GetManager gibt den internen ModelManager zurueck.
func (h *VisionHandler) GetManager() *ModelManager {
	return h.manager
}

// Close schliesst alle Ressourcen des Handlers.
func (h *VisionHandler) Close() error {
	if h.manager != nil {
		return h.manager.Close()
	}
	return nil
}

// ============================================================================
// Route Registration
// ============================================================================

// RegisterVisionRoutes registriert alle Vision API Endpoints.
// Verwendet Go 1.22+ Routing-Syntax mit HTTP-Methoden-Praefixen.
func RegisterVisionRoutes(mux *http.ServeMux, handler *VisionHandler) {
	// Encoding Endpoints
	mux.HandleFunc("POST /api/vision/encode", handler.HandleEncode)
	mux.HandleFunc("POST /api/vision/batch", handler.HandleBatch)
	mux.HandleFunc("POST /api/vision/similarity", handler.HandleSimilarity)

	// Model Management Endpoints
	mux.HandleFunc("GET /api/vision/models", handler.HandleListModels)
	mux.HandleFunc("POST /api/vision/load", handler.HandleLoadModel)
	mux.HandleFunc("POST /api/vision/unload", handler.HandleUnloadModel)
}

// RegisterVisionRoutesLegacy registriert Routes ohne HTTP-Methoden-Praefixe.
// Fuer Kompatibilitaet mit aelteren Go-Versionen.
func RegisterVisionRoutesLegacy(mux *http.ServeMux, handler *VisionHandler) {
	mux.HandleFunc("/api/vision/encode", handler.HandleEncode)
	mux.HandleFunc("/api/vision/batch", handler.HandleBatch)
	mux.HandleFunc("/api/vision/similarity", handler.HandleSimilarity)
	mux.HandleFunc("/api/vision/models", handler.HandleListModels)
	mux.HandleFunc("/api/vision/load", handler.HandleLoadModel)
	mux.HandleFunc("/api/vision/unload", handler.HandleUnloadModel)
}
