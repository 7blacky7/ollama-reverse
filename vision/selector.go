//go:build vision

// MODUL: selector
// ZWECK: Backend-Auswahl fuer Multi-Backend Vision System (ONNX/GGUF)
// INPUT: Modell-ID, Backend-Praeferenzen
// OUTPUT: Ausgewaehlter BackendType
// NEBENEFFEKTE: Runtime-Detection bei Initialisierung
// ABHAENGIGKEITEN: sync (Standard-Library), backend (fuer Detection)
// HINWEISE: Thread-sicher durch RWMutex, Auto-Mode waehlt bestes verfuegbares Backend

package vision

import (
	"strings"
	"sync"

	"github.com/ollama/ollama/vision/backend"
)

// ============================================================================
// BackendType - String-basierter Enum fuer Backend-Auswahl
// ============================================================================

// BackendType definiert den Typ des Vision-Backends.
type BackendType string

// Verfuegbare Backend-Typen
const (
	// BackendONNX verwendet ONNX Runtime fuer Inference
	BackendONNX BackendType = "onnx"

	// BackendGGUF verwendet llama.cpp/GGML fuer Inference
	BackendGGUF BackendType = "gguf"

	// BackendAuto waehlt automatisch das beste verfuegbare Backend
	BackendAuto BackendType = "auto"
)

// ============================================================================
// Modell-Muster fuer Backend-Zuordnung
// ============================================================================

// modelBackendPatterns ordnet Modell-Namen zu bevorzugten Backends zu.
var modelBackendPatterns = map[string]BackendType{
	"nomic":    BackendONNX, // Nomic Embed Vision laeuft optimal mit ONNX
	"siglip":   BackendGGUF, // SigLIP optimiert fuer GGUF/llama.cpp
	"clip":     BackendGGUF, // OpenAI CLIP mit GGUF
	"openclip": BackendGGUF, // OpenCLIP Varianten
	"dinov2":   BackendGGUF, // DINOv2 mit GGUF
	"evaclip":  BackendGGUF, // EVA-CLIP mit GGUF
}

// ============================================================================
// BackendSelector - Zentrale Backend-Auswahl
// ============================================================================

// BackendSelector verwaltet die Auswahl und Verfuegbarkeit von Vision-Backends.
// Thread-sicher durch RWMutex.
type BackendSelector struct {
	// priority definiert die Reihenfolge fuer Auto-Auswahl
	priority []BackendType

	// available speichert welche Backends verfuegbar sind
	available map[BackendType]bool

	// mu schuetzt alle Felder
	mu sync.RWMutex
}

// ============================================================================
// Konstruktor
// ============================================================================

// NewBackendSelector erstellt einen neuen BackendSelector.
// Erkennt automatisch verfuegbare Backends.
func NewBackendSelector() *BackendSelector {
	s := &BackendSelector{
		priority: []BackendType{BackendONNX, BackendGGUF},
		available: map[BackendType]bool{
			BackendONNX: false,
			BackendGGUF: false,
		},
	}

	// Verfuegbare Backends erkennen
	s.detectAvailableBackends()

	return s
}

// ============================================================================
// SelectBackend - Backend fuer Modell auswaehlen
// ============================================================================

// SelectBackend waehlt das optimale Backend fuer ein gegebenes Modell.
// Bei Auto-Mode wird basierend auf Modell-ID und Verfuegbarkeit gewaehlt.
func (s *BackendSelector) SelectBackend(modelID string) BackendType {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Modell-Name zu lowercase fuer Pattern-Matching
	modelLower := strings.ToLower(modelID)

	// Zuerst: Pruefe ob Modell ein bevorzugtes Backend hat
	for pattern, preferredBackend := range modelBackendPatterns {
		if strings.Contains(modelLower, pattern) {
			// Bevorzugtes Backend verfuegbar?
			if s.available[preferredBackend] {
				return preferredBackend
			}
			// Sonst Fallback auf anderes verfuegbares Backend
			break
		}
	}

	// Fallback: Waehle nach Prioritaet
	for _, bt := range s.priority {
		if s.available[bt] {
			return bt
		}
	}

	// Letzter Fallback: GGUF (immer via CPU moeglich)
	return BackendGGUF
}

// ============================================================================
// IsBackendAvailable - Backend-Verfuegbarkeit pruefen
// ============================================================================

// IsBackendAvailable prueft ob ein bestimmtes Backend verfuegbar ist.
func (s *BackendSelector) IsBackendAvailable(t BackendType) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Auto ist immer "verfuegbar" wenn mindestens ein Backend da ist
	if t == BackendAuto {
		for _, avail := range s.available {
			if avail {
				return true
			}
		}
		return false
	}

	return s.available[t]
}

// ============================================================================
// GetAvailableBackends - Liste verfuegbarer Backends
// ============================================================================

// GetAvailableBackends gibt eine Liste aller verfuegbaren Backends zurueck.
func (s *BackendSelector) GetAvailableBackends() []BackendType {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []BackendType
	for bt, avail := range s.available {
		if avail {
			result = append(result, bt)
		}
	}

	return result
}

// ============================================================================
// SetAvailable - Backend-Verfuegbarkeit setzen
// ============================================================================

// SetAvailable setzt die Verfuegbarkeit eines Backends.
// Wird fuer Tests und manuelle Konfiguration verwendet.
func (s *BackendSelector) SetAvailable(t BackendType, available bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Auto kann nicht direkt gesetzt werden
	if t == BackendAuto {
		return
	}

	s.available[t] = available
}

// ============================================================================
// SetPriority - Backend-Prioritaet setzen
// ============================================================================

// SetPriority setzt die Prioritaetsreihenfolge fuer Backend-Auswahl.
func (s *BackendSelector) SetPriority(priority []BackendType) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Filtere BackendAuto aus der Prioritaetsliste
	filtered := make([]BackendType, 0, len(priority))
	for _, bt := range priority {
		if bt != BackendAuto {
			filtered = append(filtered, bt)
		}
	}

	s.priority = filtered
}

// ============================================================================
// detectAvailableBackends - Runtime-Erkennung
// ============================================================================

// detectAvailableBackends erkennt welche Backends zur Laufzeit verfuegbar sind.
func (s *BackendSelector) detectAvailableBackends() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// GGUF ist immer verfuegbar (CPU-Fallback)
	s.available[BackendGGUF] = true

	// ONNX pruefe ueber backend-Package
	// ONNX ist verfuegbar wenn CUDA oder CPU-Backend vorhanden
	availableBackends := backend.DetectBackends()
	for _, b := range availableBackends {
		// ONNX benoetigt mindestens CPU Backend
		if b == backend.BackendCPU || b == backend.BackendCUDA {
			s.available[BackendONNX] = true
			break
		}
	}
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// String gibt die String-Repraesentation des BackendType zurueck.
func (t BackendType) String() string {
	return string(t)
}

// IsValid prueft ob der BackendType gueltig ist.
func (t BackendType) IsValid() bool {
	switch t {
	case BackendONNX, BackendGGUF, BackendAuto:
		return true
	default:
		return false
	}
}

// ParseBackendType konvertiert einen String zu BackendType.
// Gibt BackendAuto zurueck bei unbekanntem Wert.
func ParseBackendType(s string) BackendType {
	switch strings.ToLower(s) {
	case string(BackendONNX):
		return BackendONNX
	case string(BackendGGUF):
		return BackendGGUF
	case string(BackendAuto), "":
		return BackendAuto
	default:
		return BackendAuto
	}
}
