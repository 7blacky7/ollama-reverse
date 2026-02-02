//go:build vision

// MODUL: routes_vision_init
// ZWECK: Initialisiert Vision-Routes beim Server-Start
// INPUT: gin.Engine
// OUTPUT: Registrierte Vision-Endpoints
// NEBENEFFEKTE: Registriert HTTP-Routen
// ABHAENGIGKEITEN: routes_vision_gin.go
// HINWEISE: Wird automatisch mit -tags vision kompiliert

package server

import (
	"log/slog"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

// initVisionRoutes wird von GenerateRoutes aufgerufen wenn vision aktiv ist.
var initVisionRoutes = func(r *gin.Engine) {
	// Default model directory
	modelDir := os.Getenv("OLLAMA_VISION_MODELS")
	if modelDir == "" {
		home, _ := os.UserHomeDir()
		modelDir = filepath.Join(home, ".ollama", "vision-models")
	}

	handler := RegisterVisionRoutesGin(r, modelDir)
	slog.Info("Vision API aktiviert", "model_dir", modelDir)

	// Handler fuer spaeteres Cleanup speichern
	_ = handler
}
