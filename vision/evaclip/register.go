// MODUL: evaclip/register
// ZWECK: Automatische Registrierung des EVA-CLIP-Encoders in der globalen Registry
// INPUT: Keine
// OUTPUT: Keine (Seiteneffekt: Registry-Eintrag)
// NEBENEFFEKTE: Registriert "evaclip" Factory in vision.DefaultRegistry
// ABHAENGIGKEITEN: vision (RegisterToDefault), encoder.go (NewEVACLIPEncoder)
// HINWEISE: Wird automatisch durch init() beim Import ausgefuehrt

package evaclip

import (
	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Auto-Registrierung via init()
// ============================================================================

// init registriert den EVA-CLIP-Encoder automatisch beim Package-Import.
// Nach dem Import von "github.com/ollama/ollama/vision/evaclip" ist der Encoder
// unter dem Namen "evaclip" in der DefaultRegistry verfuegbar.
func init() {
	vision.RegisterToDefault("evaclip", evaclipFactory)
}

// evaclipFactory ist die Factory-Funktion fuer EVA-CLIP-Encoder.
func evaclipFactory(modelPath string, opts vision.LoadOptions) (vision.VisionEncoder, error) {
	return NewEVACLIPEncoder(modelPath, opts)
}
