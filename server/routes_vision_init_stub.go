//go:build !vision

// MODUL: routes_vision_init_stub
// ZWECK: Stub wenn Vision-Support nicht kompiliert ist
// INPUT: Keine
// OUTPUT: Keine
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Wird ohne -tags vision kompiliert

package server

import (
	"github.com/gin-gonic/gin"
)

// initVisionRoutes ist ein Stub wenn vision nicht aktiv ist.
var initVisionRoutes = func(r *gin.Engine) {
	// Vision-Support nicht kompiliert - nichts zu tun
}
