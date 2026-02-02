//go:build vision

// MODUL: types_vision_backends
// ZWECK: Request/Response Types fuer Vision Backend API Endpoints
// INPUT: Keine (Type-Definitionen)
// OUTPUT: Strukturierte Request/Response Types
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Verwendet fuer /api/vision/backends/* Endpoints

package server

// ============================================================================
// Response Types fuer Backend-Endpoints
// ============================================================================

// BackendsResponse ist die Antwort fuer GET /api/vision/backends.
// Enthaelt alle verfuegbaren Backends und deren Status.
type BackendsResponse struct {
	// Backends ist die Liste aller verfuegbaren Backend-Namen
	Backends []string `json:"backends"`

	// Active ist das aktuell aktive/bevorzugte Backend
	Active string `json:"active"`

	// GPUAvailable zeigt ob GPU-Beschleunigung verfuegbar ist
	GPUAvailable bool `json:"gpu_available"`

	// Models ordnet Backend-Namen zu unterstuetzten Modell-Typen zu
	Models map[string][]string `json:"models"`
}

// BackendStatusResponse ist die Antwort fuer GET /api/vision/backends/status.
// Enthaelt detaillierte Informationen zu einem Backend.
type BackendStatusResponse struct {
	// Backend ist der Name des Backends
	Backend string `json:"backend"`

	// Available zeigt ob das Backend verfuegbar ist
	Available bool `json:"available"`

	// GPUSupport zeigt ob das Backend GPU nutzen kann
	GPUSupport bool `json:"gpu_support"`

	// LoadedModels ist die Anzahl geladener Modelle
	LoadedModels int `json:"loaded_models"`
}

// SelectBackendRequest ist der Request fuer POST /api/vision/backends/select.
type SelectBackendRequest struct {
	// Backend ist der gewuenschte Backend-Name (onnx, gguf, auto)
	Backend string `json:"backend"`
}

// SelectBackendResponse ist die Antwort fuer POST /api/vision/backends/select.
type SelectBackendResponse struct {
	// Success zeigt ob die Auswahl erfolgreich war
	Success bool `json:"success"`

	// Backend ist das jetzt aktive Backend
	Backend string `json:"backend"`

	// Message ist eine optionale Statusmeldung
	Message string `json:"message,omitempty"`
}
