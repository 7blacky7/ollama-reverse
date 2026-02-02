//go:build vision

// MODUL: routes_vision_gin
// ZWECK: Registriert Vision API Endpoints im Gin-Router
// INPUT: gin.Engine, modelDir
// OUTPUT: Konfigurierte Vision-Endpoints
// NEBENEFFEKTE: Registriert HTTP-Routen in Gin
// ABHAENGIGKEITEN: gin-gonic/gin, VisionHandler, HFVisionHandler, BackendHandler
// HINWEISE: Integration der Vision API in den Haupt-Server

package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// VisionHandlers enthaelt alle Vision-Handler fuer externe Zugriffe.
type VisionHandlers struct {
	Vision  *VisionHandler
	HF      *HFVisionHandler
	Backend *BackendHandler
}

// RegisterVisionRoutesGin registriert alle Vision API Endpoints im Gin-Router.
// Wird von GenerateRoutes aufgerufen wenn das vision build tag aktiv ist.
func RegisterVisionRoutesGin(r *gin.Engine, modelDir string) (*VisionHandler, *HFVisionHandler) {
	handler := NewVisionHandler(modelDir)
	hfHandler := NewHFVisionHandler(modelDir)
	backendHandler := NewBackendHandler(handler)

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

	// HuggingFace Endpoints
	r.POST("/api/vision/load/hf", ginWrapVision(hfHandler.handleLoadHFModel))
	r.GET("/api/vision/models/hf", ginWrapVision(hfHandler.handleListHFModels))
	r.GET("/api/vision/cache", ginWrapVision(hfHandler.handleCacheStatus))
	r.DELETE("/api/vision/cache", ginWrapVision(hfHandler.handleClearCache))

	// Backend Management Endpoints
	r.GET("/api/vision/backends", ginWrapVision(backendHandler.handleListBackends))
	r.GET("/api/vision/backends/status", ginWrapVision(backendHandler.handleBackendStatus))
	r.POST("/api/vision/backends/select", ginWrapVision(backendHandler.handleSelectBackend))

	return handler, hfHandler
}

// ginWrapVision konvertiert einen http.HandlerFunc zu einem gin.HandlerFunc.
func ginWrapVision(h func(w http.ResponseWriter, r *http.Request)) gin.HandlerFunc {
	return func(c *gin.Context) {
		h(c.Writer, c.Request)
	}
}
