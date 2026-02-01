// MODUL: clip/register
// ZWECK: Automatische Registrierung des CLIP-Encoders in der globalen Registry
// INPUT: Keine
// OUTPUT: Keine (Seiteneffekt: Registry-Eintrag)
// NEBENEFFEKTE: Registriert "clip" Factory in vision.DefaultRegistry
// ABHAENGIGKEITEN: vision (RegisterToDefault), encoder.go (NewCLIPEncoder)
// HINWEISE: Wird automatisch durch init() beim Import ausgefuehrt

package clip

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Auto-Registrierung via init()
// ============================================================================

// init registriert den CLIP-Encoder automatisch beim Package-Import.
// Nach dem Import von "github.com/ollama/ollama/vision/clip" ist der Encoder
// unter dem Namen "clip" in der DefaultRegistry verfuegbar.
func init() {
	vision.RegisterToDefault("clip", clipFactory)
}

// clipFactory ist die Factory-Funktion fuer CLIP-Encoder.
func clipFactory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewCLIPEncoder(modelPath, opts)
}
