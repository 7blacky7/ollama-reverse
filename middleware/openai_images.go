// openai_images.go - Bild-Generierung Writer und Middlewares
// Implementiert OpenAI-kompatible Bildgenerierung und -bearbeitung
package middleware

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// ImageWriter verarbeitet Bildgenerierungs-Responses
type ImageWriter struct {
	BaseWriter
}

// writeResponse verarbeitet Generate-Responses fuer Bildgenerierung
func (w *ImageWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	if err := json.Unmarshal(data, &generateResponse); err != nil {
		return 0, err
	}

	// Only write response when done with image
	if generateResponse.Done && generateResponse.Image != "" {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		return len(data), json.NewEncoder(w.ResponseWriter).Encode(openai.ToImageGenerationResponse(generateResponse))
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer ImageWriter
func (w *ImageWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// ImageGenerationsMiddleware erstellt Middleware fuer Bildgenerierungs-Endpoint
func ImageGenerationsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ImageGenerationRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(openai.FromImageGenerationRequest(req)); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w
		c.Next()
	}
}

// ImageEditsMiddleware erstellt Middleware fuer Bildbearbeitungs-Endpoint
func ImageEditsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ImageEditRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		if req.Image == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "image is required"))
			return
		}

		genReq, err := openai.FromImageEditRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w
		c.Next()
	}
}
