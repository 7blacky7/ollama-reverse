// MODUL: openclip/register
// ZWECK: Automatische Registrierung des OpenCLIP-Encoders in der globalen Registry
// INPUT: Keine
// OUTPUT: Keine (Seiteneffekt: Registry-Eintrag)
// NEBENEFFEKTE: Registriert "openclip" Factory in vision.DefaultRegistry
// ABHAENGIGKEITEN: vision (RegisterToDefault), encoder.go (NewOpenCLIPEncoder)
// HINWEISE: Wird automatisch durch init() beim Import ausgefuehrt

package openclip

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Auto-Registrierung via init()
// ============================================================================

// init registriert den OpenCLIP-Encoder automatisch beim Package-Import.
// Nach dem Import von "github.com/ollama/ollama/vision/openclip" ist der Encoder
// unter dem Namen "openclip" in der DefaultRegistry verfuegbar.
func init() {
	vision.RegisterToDefault("openclip", openclipFactory)
}

// openclipFactory ist die Factory-Funktion fuer OpenCLIP-Encoder.
func openclipFactory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewOpenCLIPEncoder(modelPath, opts)
}
