//go:build vision

// MODUL: routes_vision_gin
// ZWECK: Registriert Vision API Endpoints im Gin-Router
// INPUT: gin.Engine, modelDir
// OUTPUT: Konfigurierte Vision-Endpoints
// NEBENEFFEKTE: Registriert HTTP-Routen in Gin
// ABHAENGIGKEITEN: gin-gonic/gin, VisionHandler
// HINWEISE: Integration der Vision API in den Haupt-Server

package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// RegisterVisionRoutesGin registriert alle Vision API Endpoints im Gin-Router.
// Wird von GenerateRoutes aufgerufen wenn das vision build tag aktiv ist.
func RegisterVisionRoutesGin(r *gin.Engine, modelDir string) *VisionHandler {
	handler := NewVisionHandler(modelDir)

	// Encoding Endpoints
	r.POST("/api/vision/encode", ginWrapVision(handler.HandleEncode))
	r.POST("/api/vision/batch", ginWrapVision(handler.HandleBatch))
	r.POST("/api/vision/similarity", ginWrapVision(handler.HandleSimilarity))

	// Model Management Endpoints
	r.GET("/api/vision/models", ginWrapVision(handler.HandleListModels))
	r.POST("/api/vision/load", ginWrapVision(handler.HandleLoadModel))
	r.POST("/api/vision/unload", ginWrapVision(handler.HandleUnloadModel))

	// Info Endpoint
	r.GET("/api/vision/info", ginWrapVision(handler.HandleInfo))

	return handler
}

// ginWrapVision konvertiert einen http.HandlerFunc zu einem gin.HandlerFunc.
func ginWrapVision(h func(w http.ResponseWriter, r *http.Request)) gin.HandlerFunc {
	return func(c *gin.Context) {
		h(c.Writer, c.Request)
	}
}
