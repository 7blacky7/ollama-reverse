//go:build siglip
// MODUL: vision_register
// ZWECK: Automatische Registrierung von SigLIP im vision.DefaultRegistry
// INPUT: Keine (init-basiert)
// OUTPUT: Registrierter "siglip" Encoder in vision.DefaultRegistry
// NEBENEFFEKTE: Registriert Factory bei Package-Import
// ABHAENGIGKEITEN: vision/registry_global.go, siglip/vision_adapter.go
// HINWEISE: Import dieses Packages registriert SigLIP automatisch

package siglip

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Automatische Registrierung via init()
// ============================================================================

// init registriert den SigLIP Encoder beim Package-Import.
// Verwendet MustRegisterToDefault fuer fail-fast bei Fehlern.
func init() {
	vision.MustRegisterToDefault("siglip", newSigLIPEncoder)
}

// ============================================================================
// Factory-Funktion fuer Registry
// ============================================================================

// newSigLIPEncoder ist die Factory-Funktion fuer die Registry.
// Wird von vision.DefaultRegistry aufgerufen um SigLIP Encoder zu erstellen.
func newSigLIPEncoder(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewSigLIPVisionEncoder(modelPath, opts)
}
