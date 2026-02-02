//go:build vision

// MODUL: handlers_vision_backends
// ZWECK: HTTP Handler Implementierungen fuer Vision Backend-Verwaltung
// INPUT: HTTP Requests (GET/POST) fuer Backend-Abfragen und Auswahl
// OUTPUT: JSON Responses mit Backend-Status, Listen und Konfiguration
// NEBENEFFEKTE: Aendert aktives Backend im BackendSelector
// ABHAENGIGKEITEN: vision (BackendSelector, BackendType), backend (Backend, DetectBackends)
// HINWEISE: Thread-sicher durch RWMutex auf BackendSelector, Types in types_vision_backends.go

package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/ollama/ollama/vision"
	"github.com/ollama/ollama/vision/backend"
)

// ============================================================================
// BackendHandler - Handler fuer Backend-Verwaltung
// ============================================================================

// BackendHandler verwaltet die Vision Backend API Endpoints.
// Verwendet vision.BackendSelector fuer Backend-Auswahl.
type BackendHandler struct {
	// selector ist der zentrale BackendSelector
	selector *vision.BackendSelector

	// activeBackend speichert das manuell gesetzte Backend
	activeBackend vision.BackendType

	// visionHandler Referenz fuer geladene Modelle
	visionHandler *VisionHandler

	// mu schuetzt activeBackend
	mu sync.RWMutex
}

// NewBackendHandler erstellt einen neuen BackendHandler.
func NewBackendHandler(visionHandler *VisionHandler) *BackendHandler {
	return &BackendHandler{
		selector:      vision.NewBackendSelector(),
		activeBackend: vision.BackendAuto,
		visionHandler: visionHandler,
	}
}

// ============================================================================
// HTTP Handler: GET /api/vision/backends - Liste aller Backends
// ============================================================================

// handleListBackends verarbeitet GET /api/vision/backends.
// Gibt alle verfuegbaren Backends und deren Status zurueck.
func (h *BackendHandler) handleListBackends(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeBackendError(w, http.StatusMethodNotAllowed, "BACKEND_METHOD_NOT_ALLOWED",
			"Methode nicht erlaubt, verwende GET")
		return
	}

	// Verfuegbare Backends sammeln
	availableBackends := h.selector.GetAvailableBackends()
	backendNames := make([]string, 0, len(availableBackends))
	for _, bt := range availableBackends {
		backendNames = append(backendNames, string(bt))
	}

	// Aktives Backend ermitteln
	h.mu.RLock()
	active := h.activeBackend
	h.mu.RUnlock()

	// Wenn Auto, dann das tatsaechlich genutzte Backend ermitteln
	activeStr := string(active)
	if active == vision.BackendAuto {
		activeStr = "auto"
	}

	// GPU-Verfuegbarkeit pruefen
	gpuAvailable := h.checkGPUAvailable()

	// Modelle pro Backend
	models := h.getModelsPerBackend()

	response := BackendsResponse{
		Backends:     backendNames,
		Active:       activeStr,
		GPUAvailable: gpuAvailable,
		Models:       models,
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// HTTP Handler: GET /api/vision/backends/status - Backend-Status
// ============================================================================

// handleBackendStatus verarbeitet GET /api/vision/backends/status.
// Gibt detaillierten Status eines Backends zurueck.
func (h *BackendHandler) handleBackendStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeBackendError(w, http.StatusMethodNotAllowed, "BACKEND_METHOD_NOT_ALLOWED",
			"Methode nicht erlaubt, verwende GET")
		return
	}

	// Backend aus Query-Parameter
	backendName := r.URL.Query().Get("backend")
	if backendName == "" {
		// Standard: aktuell aktives Backend
		h.mu.RLock()
		backendName = string(h.activeBackend)
		h.mu.RUnlock()
		if backendName == "auto" {
			backendName = "gguf" // Default fuer Status-Abfrage
		}
	}

	// Backend-Typ parsen
	bt := vision.ParseBackendType(backendName)
	if !bt.IsValid() || bt == vision.BackendAuto {
		bt = vision.BackendGGUF
	}

	// Verfuegbarkeit pruefen
	available := h.selector.IsBackendAvailable(bt)

	// GPU-Support pruefen
	gpuSupport := h.checkGPUSupportForBackend(bt)

	// Geladene Modelle zaehlen
	loadedModels := h.countLoadedModels()

	response := BackendStatusResponse{
		Backend:      string(bt),
		Available:    available,
		GPUSupport:   gpuSupport,
		LoadedModels: loadedModels,
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// HTTP Handler: POST /api/vision/backends/select - Backend auswaehlen
// ============================================================================

// handleSelectBackend verarbeitet POST /api/vision/backends/select.
// Setzt das bevorzugte Backend fuer zukuenftige Operationen.
func (h *BackendHandler) handleSelectBackend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeBackendError(w, http.StatusMethodNotAllowed, "BACKEND_METHOD_NOT_ALLOWED",
			"Methode nicht erlaubt, verwende POST")
		return
	}

	var req SelectBackendRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeBackendError(w, http.StatusBadRequest, "BACKEND_INVALID_REQUEST",
			fmt.Sprintf("Ungueltiger Request: %v", err))
		return
	}

	if req.Backend == "" {
		h.writeBackendError(w, http.StatusBadRequest, "BACKEND_INVALID_REQUEST",
			"backend ist erforderlich")
		return
	}

	// Backend-Typ parsen
	bt := vision.ParseBackendType(req.Backend)
	if !bt.IsValid() {
		h.writeBackendError(w, http.StatusBadRequest, "BACKEND_INVALID_TYPE",
			fmt.Sprintf("Ungueltiger Backend-Typ: %s (erlaubt: onnx, gguf, auto)", req.Backend))
		return
	}

	// Bei konkretem Backend: Verfuegbarkeit pruefen
	if bt != vision.BackendAuto && !h.selector.IsBackendAvailable(bt) {
		h.writeBackendError(w, http.StatusBadRequest, "BACKEND_NOT_AVAILABLE",
			fmt.Sprintf("Backend nicht verfuegbar: %s", req.Backend))
		return
	}

	// Backend setzen
	h.mu.Lock()
	h.activeBackend = bt
	h.mu.Unlock()

	// Prioritaet im Selector anpassen wenn nicht Auto
	if bt != vision.BackendAuto {
		h.selector.SetPriority([]vision.BackendType{bt})
	}

	message := fmt.Sprintf("Backend erfolgreich auf '%s' gesetzt", bt)
	if bt == vision.BackendAuto {
		message = "Automatische Backend-Auswahl aktiviert"
	}

	response := SelectBackendResponse{
		Success: true,
		Backend: string(bt),
		Message: message,
	}

	h.writeJSON(w, http.StatusOK, response)
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// checkGPUAvailable prueft ob irgendeine GPU verfuegbar ist.
func (h *BackendHandler) checkGPUAvailable() bool {
	backends := backend.DetectBackends()
	for _, b := range backends {
		if b == backend.BackendCUDA || b == backend.BackendMetal {
			return true
		}
	}
	return false
}

// checkGPUSupportForBackend prueft GPU-Support fuer ein bestimmtes Backend.
func (h *BackendHandler) checkGPUSupportForBackend(bt vision.BackendType) bool {
	if !h.selector.IsBackendAvailable(bt) {
		return false
	}

	backends := backend.DetectBackends()
	for _, b := range backends {
		if b == backend.BackendCUDA || b == backend.BackendMetal {
			return true
		}
	}
	return false
}

// countLoadedModels zaehlt die aktuell geladenen Modelle.
func (h *BackendHandler) countLoadedModels() int {
	if h.visionHandler == nil {
		return 0
	}

	h.visionHandler.mu.RLock()
	count := len(h.visionHandler.models)
	h.visionHandler.mu.RUnlock()

	return count
}

// getModelsPerBackend gibt die unterstuetzten Modell-Typen pro Backend zurueck.
func (h *BackendHandler) getModelsPerBackend() map[string][]string {
	return map[string][]string{
		"onnx": {"nomic", "nomic-embed-vision"},
		"gguf": {"siglip", "clip", "openclip", "dinov2", "evaclip"},
	}
}

// GetActiveBackend gibt das aktuell aktive Backend zurueck.
func (h *BackendHandler) GetActiveBackend() vision.BackendType {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.activeBackend
}

// GetSelector gibt den BackendSelector zurueck.
func (h *BackendHandler) GetSelector() *vision.BackendSelector {
	return h.selector
}

// ============================================================================
// JSON Response Helper
// ============================================================================

// writeJSON schreibt eine JSON-Response.
func (h *BackendHandler) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeBackendError schreibt eine Fehler-Response.
func (h *BackendHandler) writeBackendError(w http.ResponseWriter, status int, code, message string) {
	h.writeJSON(w, status, map[string]string{
		"code":    code,
		"message": message,
	})
}
