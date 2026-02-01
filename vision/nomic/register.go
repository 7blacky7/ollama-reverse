// MODUL: register
// ZWECK: Registriert Nomic Encoder in der globalen vision.DefaultRegistry
// INPUT: Keine (init()-Funktion)
// OUTPUT: Keine
// NEBENEFFEKTE: Modifiziert vision.DefaultRegistry beim Package-Import
// ABHAENGIGKEITEN: vision (DefaultRegistry, EncoderFactory), encoder.go
// HINWEISE: Import des Packages registriert automatisch den Encoder
//           Beispiel: import _ "github.com/ollama/ollama/vision/nomic"

package nomic

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Package Initialisierung
// ============================================================================

// init registriert den Nomic Encoder in der DefaultRegistry.
// Wird automatisch beim Import des Packages ausgefuehrt.
func init() {
	vision.MustRegisterToDefault(EncoderName, factory)
}

// ============================================================================
// Factory Funktion
// ============================================================================

// factory erstellt einen neuen NomicEncoder.
// Implementiert vision.EncoderFactory Signatur.
func factory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewNomicEncoder(modelPath, opts)
}
