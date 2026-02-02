//go:build vision

// MODUL: registry_backend
// ZWECK: Backend-spezifische Registry-Erweiterungen fuer Multi-Backend System
// INPUT: Modell-ID, BackendType, EncoderFactory
// OUTPUT: VisionEncoder fuer spezifisches Backend
// NEBENEFFEKTE: Keine (rein speicherbasiert)
// ABHAENGIGKEITEN: registry.go, selector.go, factory.go
// HINWEISE: Thread-sicher, unterstuetzt GGUF/ONNX Fallback-Kette

package vision

import (
	"errors"
	"sync"
)

// ============================================================================
// Fehler-Definitionen fuer Backend-Registry
// ============================================================================

var (
	// ErrNoBackendAvailable wenn kein Backend verfuegbar ist
	ErrNoBackendAvailable = errors.New("vision: kein backend verfuegbar")

	// ErrBackendNotSupported wenn Backend fuer Modell nicht unterstuetzt wird
	ErrBackendNotSupported = errors.New("vision: backend nicht unterstuetzt")

	// ErrModelPathRequired wenn Modell-Pfad fehlt
	ErrModelPathRequired = errors.New("vision: modell-pfad erforderlich")
)

// ============================================================================
// BackendRegistry - Erweiterte Registry mit Backend-Unterstuetzung
// ============================================================================

// BackendRegistry erweitert Registry um Backend-spezifische Verwaltung.
// Ermoeglicht Registrierung von Factories pro Backend-Typ.
type BackendRegistry struct {
	*Registry

	// backendFactories speichert Factories nach Backend-Typ
	backendFactories map[BackendType]map[string]EncoderFactory

	// selector fuer automatische Backend-Auswahl
	selector *BackendSelector

	// mu schuetzt backendFactories
	bmu sync.RWMutex
}

// ============================================================================
// Konstruktor
// ============================================================================

// NewBackendRegistry erstellt eine neue BackendRegistry.
// Integriert automatisch einen BackendSelector fuer Auto-Auswahl.
func NewBackendRegistry() *BackendRegistry {
	return &BackendRegistry{
		Registry: NewRegistry(),
		backendFactories: map[BackendType]map[string]EncoderFactory{
			BackendGGUF: make(map[string]EncoderFactory),
			BackendONNX: make(map[string]EncoderFactory),
		},
		selector: NewBackendSelector(),
	}
}

// ============================================================================
// Backend-spezifische Registrierung
// ============================================================================

// RegisterForBackend registriert eine Factory fuer ein spezifisches Backend.
// name: Encoder-Name (z.B. "clip", "nomic")
// backend: Backend-Typ (BackendGGUF oder BackendONNX)
// factory: EncoderFactory-Funktion
func (r *BackendRegistry) RegisterForBackend(name string, backend BackendType, factory EncoderFactory) {
	r.bmu.Lock()
	defer r.bmu.Unlock()

	// BackendAuto wird nicht direkt registriert
	if backend == BackendAuto {
		return
	}

	// Map initialisieren falls noetig
	if r.backendFactories[backend] == nil {
		r.backendFactories[backend] = make(map[string]EncoderFactory)
	}

	r.backendFactories[backend][name] = factory
}

// UnregisterForBackend entfernt eine Factory fuer ein spezifisches Backend.
// Gibt true zurueck wenn die Factory existierte, sonst false.
func (r *BackendRegistry) UnregisterForBackend(name string, backend BackendType) bool {
	r.bmu.Lock()
	defer r.bmu.Unlock()

	if backend == BackendAuto {
		return false
	}

	factories, exists := r.backendFactories[backend]
	if !exists {
		return false
	}

	_, found := factories[name]
	if found {
		delete(factories, name)
	}
	return found
}

// ============================================================================
// Backend-spezifische Abfragen
// ============================================================================

// GetFactoriesForBackend gibt eine Liste aller Encoder-Namen fuer ein Backend zurueck.
func (r *BackendRegistry) GetFactoriesForBackend(backend BackendType) []string {
	r.bmu.RLock()
	defer r.bmu.RUnlock()

	// Bei Auto alle kombinieren (ohne Duplikate)
	if backend == BackendAuto {
		seen := make(map[string]bool)
		var result []string

		for _, factories := range r.backendFactories {
			for name := range factories {
				if !seen[name] {
					seen[name] = true
					result = append(result, name)
				}
			}
		}
		return result
	}

	factories, exists := r.backendFactories[backend]
	if !exists {
		return nil
	}

	names := make([]string, 0, len(factories))
	for name := range factories {
		names = append(names, name)
	}
	return names
}

// HasForBackend prueft ob ein Encoder fuer ein bestimmtes Backend registriert ist.
func (r *BackendRegistry) HasForBackend(name string, backend BackendType) bool {
	r.bmu.RLock()
	defer r.bmu.RUnlock()

	factories, exists := r.backendFactories[backend]
	if !exists {
		return false
	}

	_, found := factories[name]
	return found
}

// GetForBackend gibt die Factory fuer Name und Backend zurueck.
func (r *BackendRegistry) GetForBackend(name string, backend BackendType) (EncoderFactory, bool) {
	r.bmu.RLock()
	defer r.bmu.RUnlock()

	factories, exists := r.backendFactories[backend]
	if !exists {
		return nil, false
	}

	factory, found := factories[name]
	return factory, found
}

// ============================================================================
// Selector-Zugriff
// ============================================================================

// Selector gibt den internen BackendSelector zurueck.
// Ermoeglicht direkte Backend-Abfragen und -Konfiguration.
func (r *BackendRegistry) Selector() *BackendSelector {
	return r.selector
}

// SetSelector setzt einen neuen BackendSelector.
// Nuetzlich fuer Tests mit Mock-Selectors.
func (r *BackendRegistry) SetSelector(selector *BackendSelector) {
	r.bmu.Lock()
	defer r.bmu.Unlock()
	r.selector = selector
}

// ============================================================================
// Statistik-Methoden
// ============================================================================

// BackendStats gibt Statistiken ueber registrierte Factories pro Backend zurueck.
func (r *BackendRegistry) BackendStats() map[BackendType]int {
	r.bmu.RLock()
	defer r.bmu.RUnlock()

	stats := make(map[BackendType]int)
	for backend, factories := range r.backendFactories {
		stats[backend] = len(factories)
	}
	return stats
}
