//go:build vision

// MODUL: registry_backend_global
// ZWECK: Globale BackendRegistry-Instanz und Package-Level Convenience-Funktionen
// INPUT: Modell-ID, BackendType, EncoderFactory, LoadOptions
// OUTPUT: VisionEncoder-Instanzen
// NEBENEFFEKTE: Aendert globale DefaultBackendRegistry
// ABHAENGIGKEITEN: registry_backend.go (BackendRegistry)
// HINWEISE: Encoder koennen via init() mit Backend-Info registriert werden

package vision

// ============================================================================
// Globale Backend-Registry Instanz
// ============================================================================

// DefaultBackendRegistry ist die globale BackendRegistry fuer Vision-Encoder.
// Unterstuetzt Backend-spezifische Registrierung und automatische Auswahl.
var DefaultBackendRegistry = NewBackendRegistry()

// ============================================================================
// Package-Level Convenience-Funktionen - Registrierung
// ============================================================================

// RegisterEncoderForBackend registriert eine Factory in der DefaultBackendRegistry.
// name: Encoder-Name (z.B. "clip", "nomic")
// backend: Backend-Typ (BackendGGUF oder BackendONNX)
// factory: EncoderFactory-Funktion
func RegisterEncoderForBackend(name string, backend BackendType, factory EncoderFactory) {
	DefaultBackendRegistry.RegisterForBackend(name, backend, factory)
}

// UnregisterEncoderForBackend entfernt eine Factory aus der DefaultBackendRegistry.
func UnregisterEncoderForBackend(name string, backend BackendType) bool {
	return DefaultBackendRegistry.UnregisterForBackend(name, backend)
}

// MustRegisterEncoderForBackend registriert eine Factory und panict bei nil.
// Nuetzlich fuer init()-Funktionen wo Fehler fatal sein sollten.
func MustRegisterEncoderForBackend(name string, backend BackendType, factory EncoderFactory) {
	if factory == nil {
		panic("vision: nil factory fuer encoder '" + name + "' backend " + string(backend))
	}
	RegisterEncoderForBackend(name, backend, factory)
}

// ============================================================================
// Package-Level Convenience-Funktionen - Abfrage
// ============================================================================

// GetEncoderForBackend gibt die Factory aus der DefaultBackendRegistry zurueck.
func GetEncoderForBackend(name string, backend BackendType) (EncoderFactory, bool) {
	return DefaultBackendRegistry.GetForBackend(name, backend)
}

// HasEncoderForBackend prueft ob Encoder fuer Backend registriert ist.
func HasEncoderForBackend(name string, backend BackendType) bool {
	return DefaultBackendRegistry.HasForBackend(name, backend)
}

// ListEncodersForBackend gibt alle Encoder-Namen fuer ein Backend zurueck.
func ListEncodersForBackend(backend BackendType) []string {
	return DefaultBackendRegistry.GetFactoriesForBackend(backend)
}

// GetBackendStats gibt Statistiken ueber registrierte Encoder zurueck.
func GetBackendStats() map[BackendType]int {
	return DefaultBackendRegistry.BackendStats()
}

// ============================================================================
// Package-Level Convenience-Funktionen - Erstellung
// ============================================================================

// CreateEncoderWithBackend erstellt einen Encoder mit spezifischem Backend.
// modelID: Modell-Identifikator (z.B. "clip", "nomic")
// backend: Gewuenschtes Backend (GGUF, ONNX)
// modelPath: Pfad zur Modell-Datei
// opts: LoadOptions fuer Encoder-Konfiguration
func CreateEncoderWithBackend(
	modelID string,
	backend BackendType,
	modelPath string,
	opts LoadOptions,
) (VisionEncoder, error) {
	return DefaultBackendRegistry.CreateEncoder(modelID, backend, modelPath, opts)
}

// CreateEncoderAutomatic erstellt einen Encoder mit automatischer Backend-Auswahl.
// Implementiert Fallback-Kette basierend auf BackendSelector.
// modelID: Modell-Identifikator (z.B. "clip", "nomic")
// modelPath: Pfad zur Modell-Datei
// opts: LoadOptions fuer Encoder-Konfiguration
func CreateEncoderAutomatic(
	modelID string,
	modelPath string,
	opts LoadOptions,
) (VisionEncoder, error) {
	return DefaultBackendRegistry.CreateEncoderAuto(modelID, modelPath, opts)
}

// CreateEncoderAutomaticWithDefaults erstellt Encoder mit Auto-Backend und Defaults.
// Convenience-Methode fuer einfache Faelle.
func CreateEncoderAutomaticWithDefaults(modelID string, modelPath string) (VisionEncoder, error) {
	return DefaultBackendRegistry.CreateEncoderAuto(modelID, modelPath, DefaultLoadOptions())
}

// ============================================================================
// Backend-Selector Zugriff
// ============================================================================

// GetDefaultBackendSelector gibt den BackendSelector der DefaultBackendRegistry zurueck.
func GetDefaultBackendSelector() *BackendSelector {
	return DefaultBackendRegistry.Selector()
}

// GetAvailableBackends gibt alle verfuegbaren Backends zurueck.
func GetAvailableBackends() []BackendType {
	return DefaultBackendRegistry.Selector().GetAvailableBackends()
}

// IsBackendAvailable prueft ob ein Backend verfuegbar ist.
func IsBackendAvailable(backend BackendType) bool {
	return DefaultBackendRegistry.Selector().IsBackendAvailable(backend)
}

// SelectBestBackend waehlt das beste Backend fuer ein Modell.
func SelectBestBackend(modelID string) BackendType {
	return DefaultBackendRegistry.Selector().SelectBackend(modelID)
}
