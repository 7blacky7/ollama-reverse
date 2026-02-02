//go:build vision

// MODUL: router_vision
// ZWECK: Registriert Vision API Endpoints und definiert VisionHandler
// INPUT: http.ServeMux, VisionHandler
// OUTPUT: Konfigurierter HTTP-Router
// NEBENEFFEKTE: Registriert HTTP-Routen
// ABHAENGIGKEITEN: vision (VisionEncoder, Registry)
// HINWEISE: Verwendet vision.Registry fuer Encoder-Verwaltung

package server

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Vision Encoder Interfaces - fuer Handler-Entkopplung
// ============================================================================

// VisionEncoderInterface definiert die Methoden die ein Vision Encoder haben muss.
// Dieses Interface spiegelt vision.VisionEncoder fuer interne Verwendung.
type VisionEncoderInterface interface {
	Encode(imageData []byte) ([]float32, error)
	EncodeBatch(images [][]byte) ([][]float32, error)
	Close() error
	ModelInfo() vision.ModelInfo
}

// VisionRegistryInterface definiert den Zugriff auf die Encoder-Registry.
type VisionRegistryInterface interface {
	Has(name string) bool
	List() []string
	Create(name string, modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error)
	CreateWithDefaults(name string, modelPath string) (vision.VisionEncoder, error)
}

// ============================================================================
// Vision Handler
// ============================================================================

// VisionHandler verwaltet die Vision API Endpoints.
// Verwendet vision.Registry fuer Encoder-Typen und eigenen Cache fuer geladene Modelle.
type VisionHandler struct {
	// registry ist die Encoder-Factory-Registry
	registry *vision.Registry

	// models speichert geladene Encoder nach Namen
	models map[string]vision.VisionEncoder

	// paths speichert die Pfade zu den geladenen Modellen
	paths map[string]string

	// mu schuetzt models und paths
	mu sync.RWMutex

	// modelDir ist das Default-Verzeichnis fuer Modelle
	modelDir string
}

// NewVisionHandler erstellt einen neuen VisionHandler.
// modelDir ist das Verzeichnis fuer Modell-Dateien.
func NewVisionHandler(modelDir string) *VisionHandler {
	return &VisionHandler{
		registry: vision.DefaultRegistry,
		models:   make(map[string]vision.VisionEncoder),
		paths:    make(map[string]string),
		modelDir: modelDir,
	}
}

// NewVisionHandlerWithRegistry erstellt einen VisionHandler mit eigener Registry.
// Ermoeglicht das Testen mit Mock-Registry.
func NewVisionHandlerWithRegistry(registry *vision.Registry, modelDir string) *VisionHandler {
	return &VisionHandler{
		registry: registry,
		models:   make(map[string]vision.VisionEncoder),
		paths:    make(map[string]string),
		modelDir: modelDir,
	}
}

// GetRegistry gibt die interne Registry zurueck.
func (h *VisionHandler) GetRegistry() *vision.Registry {
	return h.registry
}

// getModel holt ein geladenes Modell aus dem Cache.
// Gibt einen Fehler zurueck wenn das Modell nicht geladen ist.
func (h *VisionHandler) getModel(name string) (vision.VisionEncoder, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	encoder, exists := h.models[name]
	if !exists {
		return nil, errModelNotLoaded(name)
	}
	return encoder, nil
}

// Close schliesst alle geladenen Modelle und gibt Ressourcen frei.
func (h *VisionHandler) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	var firstErr error
	for name, encoder := range h.models {
		if err := encoder.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(h.models, name)
		delete(h.paths, name)
	}
	return firstErr
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

// ============================================================================
// HTTP Response Helpers - Methoden fuer VisionHandler
// ============================================================================

// writeVisionJSON schreibt eine JSON-Response.
func (h *VisionHandler) writeVisionJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeVisionError schreibt eine strukturierte Fehler-Response.
func (h *VisionHandler) writeVisionError(w http.ResponseWriter, status int, message, code string) {
	h.writeVisionJSON(w, status, VisionAPIError{
		Code:    code,
		Message: message,
	})
}

// decodeVisionJSON dekodiert JSON aus dem Request-Body.
func decodeVisionJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}
