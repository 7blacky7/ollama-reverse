// Package vision - Encoder Registry fuer dynamische Modell-Registrierung.
//
// MODUL: registry
// ZWECK: Zentrale Registry fuer Vision-Encoder-Factories mit Thread-sicherer Verwaltung
// INPUT: Encoder-Name, EncoderFactory-Funktionen, LoadOptions
// OUTPUT: Registrierte Encoder-Instanzen
// NEBENEFFEKTE: Keine (rein speicherbasiert)
// ABHAENGIGKEITEN: sync (stdlib), factory.go (EncoderFactory, VisionEncoder)
// HINWEISE: Thread-sicher durch RWMutex, nutzt EncoderFactory aus factory.go
package vision

import (
	"sync"
)

// ============================================================================
// Registry - Zentrale Encoder-Verwaltung
// ============================================================================

// Registry verwaltet registrierte Vision-Encoder-Factories.
// Thread-sicher durch RWMutex.
// Verwendet EncoderFactory aus factory.go.
type Registry struct {
	encoders map[string]EncoderFactory
	mu       sync.RWMutex
}

// NewRegistry erstellt eine neue leere Registry.
func NewRegistry() *Registry {
	return &Registry{
		encoders: make(map[string]EncoderFactory),
	}
}

// ============================================================================
// Registry Methoden - Registrierung
// ============================================================================

// Register registriert eine neue EncoderFactory unter dem angegebenen Namen.
// Ueberschreibt existierende Eintraege ohne Warnung.
// factory: Funktion vom Typ func(modelPath string, opts LoadOptions) (VisionEncoder, error)
func (r *Registry) Register(name string, factory EncoderFactory) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.encoders[name] = factory
}

// Unregister entfernt einen Encoder aus der Registry.
// Gibt true zurueck wenn der Encoder existierte, sonst false.
func (r *Registry) Unregister(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	_, exists := r.encoders[name]
	if exists {
		delete(r.encoders, name)
	}
	return exists
}

// ============================================================================
// Registry Methoden - Abfrage
// ============================================================================

// Get gibt die Factory fuer den angegebenen Namen zurueck.
// Gibt (factory, true) wenn gefunden, sonst (nil, false).
func (r *Registry) Get(name string) (EncoderFactory, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	factory, exists := r.encoders[name]
	return factory, exists
}

// Has prueft ob ein Encoder unter dem Namen registriert ist.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, exists := r.encoders[name]
	return exists
}

// List gibt eine Liste aller registrierten Encoder-Namen zurueck.
// Die Reihenfolge ist nicht deterministisch.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.encoders))
	for name := range r.encoders {
		names = append(names, name)
	}
	return names
}

// Count gibt die Anzahl registrierter Encoder zurueck.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.encoders)
}

// ============================================================================
// Registry Methoden - Encoder-Erstellung
// ============================================================================

// Create erstellt einen neuen Encoder mit der registrierten Factory.
// Gibt ErrEncoderNotRegistered zurueck wenn der Name nicht gefunden wurde.
// modelPath: Pfad zur GGUF-Modelldatei
// opts: LoadOptions fuer die Encoder-Konfiguration
func (r *Registry) Create(name string, modelPath string, opts LoadOptions) (VisionEncoder, error) {
	factory, exists := r.Get(name)
	if !exists {
		return nil, &RegistryError{
			Op:   "create",
			Name: name,
			Err:  ErrEncoderNotRegistered,
		}
	}

	return factory(modelPath, opts)
}

// CreateWithDefaults erstellt einen Encoder mit Default-Optionen.
// Convenience-Methode fuer einfache Faelle.
func (r *Registry) CreateWithDefaults(name string, modelPath string) (VisionEncoder, error) {
	return r.Create(name, modelPath, DefaultLoadOptions())
}
