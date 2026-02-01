// Package vision - Globale Registry-Instanz und Package-Level Funktionen.
//
// MODUL: registry_global
// ZWECK: Stellt eine globale DefaultRegistry bereit und Package-Funktionen als Convenience-Wrapper
// INPUT: Encoder-Name, EncoderFactory, LoadOptions
// OUTPUT: Registrierte Encoder-Instanzen
// NEBENEFFEKTE: Aendert globale DefaultRegistry
// ABHAENGIGKEITEN: registry.go (Registry), factory.go (EncoderFactory)
// HINWEISE: Encoder koennen via init() in ihren Packages automatisch registriert werden
package vision

import "errors"

// ============================================================================
// Registry Errors - Fehlercodes fuer Registry-Operationen
// ============================================================================

// ErrEncoderNotRegistered wird zurueckgegeben wenn ein Encoder nicht registriert ist.
var ErrEncoderNotRegistered = errors.New("vision: encoder not registered")

// RegistryError repraesentiert einen Registry-spezifischen Fehler.
type RegistryError struct {
	Op   string // Operation (z.B. "create", "get")
	Name string // Encoder-Name
	Err  error  // Urspruenglicher Fehler
}

// Error implementiert das error Interface.
func (e *RegistryError) Error() string {
	return "vision: " + e.Op + " encoder '" + e.Name + "': " + e.Err.Error()
}

// Unwrap gibt den urspruenglichen Fehler zurueck.
func (e *RegistryError) Unwrap() error {
	return e.Err
}

// ============================================================================
// Globale Registry-Instanz
// ============================================================================

// DefaultRegistry ist die globale Registry fuer Vision-Encoder.
// Encoder registrieren sich typischerweise via init() in ihren Packages.
var DefaultRegistry = NewRegistry()

// ============================================================================
// Package-Level Convenience-Funktionen - Registrierung
// ============================================================================

// RegisterToDefault registriert eine EncoderFactory in der DefaultRegistry.
// Wrapper fuer DefaultRegistry.Register().
// Hinweis: RegisterEncoder in factory.go nutzt eine separate globale Registry.
func RegisterToDefault(name string, factory EncoderFactory) {
	DefaultRegistry.Register(name, factory)
}

// UnregisterFromDefault entfernt einen Encoder aus der DefaultRegistry.
// Wrapper fuer DefaultRegistry.Unregister().
func UnregisterFromDefault(name string) bool {
	return DefaultRegistry.Unregister(name)
}

// ============================================================================
// Package-Level Convenience-Funktionen - Abfrage
// ============================================================================

// GetFromDefault gibt die Factory fuer den Namen aus der DefaultRegistry zurueck.
// Wrapper fuer DefaultRegistry.Get().
func GetFromDefault(name string) (EncoderFactory, bool) {
	return DefaultRegistry.Get(name)
}

// HasInDefault prueft ob ein Encoder in der DefaultRegistry registriert ist.
// Wrapper fuer DefaultRegistry.Has().
func HasInDefault(name string) bool {
	return DefaultRegistry.Has(name)
}

// ListFromDefault gibt alle registrierten Encoder-Namen aus der DefaultRegistry zurueck.
// Wrapper fuer DefaultRegistry.List().
func ListFromDefault() []string {
	return DefaultRegistry.List()
}

// CountInDefault gibt die Anzahl registrierter Encoder in der DefaultRegistry zurueck.
// Wrapper fuer DefaultRegistry.Count().
func CountInDefault() int {
	return DefaultRegistry.Count()
}

// ============================================================================
// Package-Level Convenience-Funktionen - Erstellung
// ============================================================================

// CreateFromDefault erstellt einen Encoder mit der DefaultRegistry.
// Wrapper fuer DefaultRegistry.Create().
func CreateFromDefault(name string, modelPath string, opts LoadOptions) (VisionEncoder, error) {
	return DefaultRegistry.Create(name, modelPath, opts)
}

// CreateFromDefaultWithDefaults erstellt einen Encoder mit Default-Optionen.
// Wrapper fuer DefaultRegistry.CreateWithDefaults().
func CreateFromDefaultWithDefaults(name string, modelPath string) (VisionEncoder, error) {
	return DefaultRegistry.CreateWithDefaults(name, modelPath)
}

// ============================================================================
// Registrierungs-Helfer
// ============================================================================

// MustRegisterToDefault registriert eine Factory und panict bei nil-Factory.
// Nuetzlich fuer init()-Funktionen wo Fehler fatal sein sollten.
func MustRegisterToDefault(name string, factory EncoderFactory) {
	if factory == nil {
		panic("vision: nil factory for encoder '" + name + "'")
	}
	RegisterToDefault(name, factory)
}
