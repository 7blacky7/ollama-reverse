//go:build vision

// MODUL: registry_backend_create
// ZWECK: Encoder-Erstellung mit Backend-Auswahl und Fallback-Kette
// INPUT: Modell-ID, BackendType, Modell-Pfad, LoadOptions
// OUTPUT: VisionEncoder-Instanzen
// NEBENEFFEKTE: Laedt Modelle ueber Factory-Funktionen
// ABHAENGIGKEITEN: registry_backend.go (BackendRegistry)
// HINWEISE: Implementiert GGUF -> ONNX Fallback-Kette

package vision

import (
	"fmt"
)

// ============================================================================
// Encoder-Erstellung mit Backend-Auswahl
// ============================================================================

// CreateEncoder erstellt einen Encoder fuer ein spezifisches Backend.
// modelID: Modell-Identifikator (z.B. "clip", "nomic")
// backend: Gewuenschtes Backend (GGUF, ONNX)
// modelPath: Pfad zur Modell-Datei
// opts: LoadOptions fuer Encoder-Konfiguration
func (r *BackendRegistry) CreateEncoder(
	modelID string,
	backend BackendType,
	modelPath string,
	opts LoadOptions,
) (VisionEncoder, error) {
	// Pfad validieren
	if modelPath == "" {
		return nil, &RegistryError{
			Op:   "create_encoder",
			Name: modelID,
			Err:  ErrModelPathRequired,
		}
	}

	// Backend-Verfuegbarkeit pruefen
	if !r.selector.IsBackendAvailable(backend) {
		return nil, &RegistryError{
			Op:   "create_encoder",
			Name: modelID,
			Err:  fmt.Errorf("%w: %s", ErrBackendNotSupported, backend),
		}
	}

	// Factory fuer Backend holen
	factory, exists := r.GetForBackend(modelID, backend)
	if !exists {
		return nil, &RegistryError{
			Op:   "create_encoder",
			Name: modelID,
			Err:  fmt.Errorf("%w fuer backend %s", ErrEncoderNotRegistered, backend),
		}
	}

	return factory(modelPath, opts)
}

// CreateEncoderAuto erstellt einen Encoder mit automatischer Backend-Auswahl.
// Implementiert Fallback-Kette: GGUF -> ONNX
// modelID: Modell-Identifikator (z.B. "clip", "nomic")
// modelPath: Pfad zur Modell-Datei
// opts: LoadOptions fuer Encoder-Konfiguration
func (r *BackendRegistry) CreateEncoderAuto(
	modelID string,
	modelPath string,
	opts LoadOptions,
) (VisionEncoder, error) {
	// Pfad validieren
	if modelPath == "" {
		return nil, &RegistryError{
			Op:   "create_encoder_auto",
			Name: modelID,
			Err:  ErrModelPathRequired,
		}
	}

	// Verfuegbare Backends holen
	availableBackends := r.selector.GetAvailableBackends()
	if len(availableBackends) == 0 {
		return nil, &RegistryError{
			Op:   "create_encoder_auto",
			Name: modelID,
			Err:  ErrNoBackendAvailable,
		}
	}

	// Optimales Backend via Selector bestimmen
	selectedBackend := r.selector.SelectBackend(modelID)

	// Fallback-Kette: Versuche zuerst das ausgewaehlte Backend
	// dann die restlichen in Prioritaetsreihenfolge
	backendOrder := r.buildFallbackOrder(selectedBackend)

	var lastErr error
	for _, backend := range backendOrder {
		// Backend verfuegbar?
		if !r.selector.IsBackendAvailable(backend) {
			continue
		}

		// Factory registriert?
		factory, exists := r.GetForBackend(modelID, backend)
		if !exists {
			continue
		}

		// Encoder erstellen
		encoder, err := factory(modelPath, opts)
		if err == nil {
			return encoder, nil
		}
		lastErr = err
	}

	// Kein Backend konnte Encoder erstellen
	if lastErr != nil {
		return nil, &RegistryError{
			Op:   "create_encoder_auto",
			Name: modelID,
			Err:  lastErr,
		}
	}

	return nil, &RegistryError{
		Op:   "create_encoder_auto",
		Name: modelID,
		Err:  fmt.Errorf("%w: kein backend unterstuetzt modell", ErrNoBackendAvailable),
	}
}

// ============================================================================
// Fallback-Hilfsfunktionen
// ============================================================================

// buildFallbackOrder erstellt die Backend-Reihenfolge fuer Fallback.
// Beginnt mit dem uebergebenen Backend, dann GGUF vor ONNX.
func (r *BackendRegistry) buildFallbackOrder(preferred BackendType) []BackendType {
	// Standard-Prioritaet: GGUF (CPU-Fallback immer moeglich) -> ONNX
	order := []BackendType{BackendGGUF, BackendONNX}

	// Wenn preferred nicht GGUF, stelle es an den Anfang
	if preferred == BackendONNX {
		order = []BackendType{BackendONNX, BackendGGUF}
	}

	return order
}

// CreateEncoderWithFallback erstellt einen Encoder mit expliziter Fallback-Kette.
// Versucht Backends in der uebergebenen Reihenfolge.
// modelID: Modell-Identifikator
// modelPath: Pfad zur Modell-Datei
// opts: LoadOptions
// fallbackChain: Reihenfolge der Backends (z.B. [BackendGGUF, BackendONNX])
func (r *BackendRegistry) CreateEncoderWithFallback(
	modelID string,
	modelPath string,
	opts LoadOptions,
	fallbackChain []BackendType,
) (VisionEncoder, error) {
	// Pfad validieren
	if modelPath == "" {
		return nil, &RegistryError{
			Op:   "create_encoder_fallback",
			Name: modelID,
			Err:  ErrModelPathRequired,
		}
	}

	if len(fallbackChain) == 0 {
		return nil, &RegistryError{
			Op:   "create_encoder_fallback",
			Name: modelID,
			Err:  ErrNoBackendAvailable,
		}
	}

	var lastErr error
	for _, backend := range fallbackChain {
		// BackendAuto ueberspringen
		if backend == BackendAuto {
			continue
		}

		// Backend verfuegbar?
		if !r.selector.IsBackendAvailable(backend) {
			continue
		}

		// Factory registriert?
		factory, exists := r.GetForBackend(modelID, backend)
		if !exists {
			continue
		}

		// Encoder erstellen
		encoder, err := factory(modelPath, opts)
		if err == nil {
			return encoder, nil
		}
		lastErr = err
	}

	// Kein Backend erfolgreich
	if lastErr != nil {
		return nil, &RegistryError{
			Op:   "create_encoder_fallback",
			Name: modelID,
			Err:  lastErr,
		}
	}

	return nil, &RegistryError{
		Op:   "create_encoder_fallback",
		Name: modelID,
		Err:  fmt.Errorf("%w: kein backend in fallback-kette unterstuetzt modell", ErrNoBackendAvailable),
	}
}
