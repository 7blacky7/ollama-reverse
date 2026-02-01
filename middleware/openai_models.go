// openai_models.go - Model-bezogene Writer und Middlewares
// Verarbeitet List-, Retrieve- und Embed-Operationen
package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// writeResponse verarbeitet List-Responses und konvertiert sie ins OpenAI-Format
func (w *ListWriter) writeResponse(data []byte) (int, error) {
	var listResponse api.ListResponse
	err := json.Unmarshal(data, &listResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToListCompletion(listResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer ListWriter
func (w *ListWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// writeResponse verarbeitet Show-Responses und konvertiert sie ins OpenAI-Format
func (w *RetrieveWriter) writeResponse(data []byte) (int, error) {
	var showResponse api.ShowResponse
	err := json.Unmarshal(data, &showResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToModel(showResponse, w.model))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer RetrieveWriter
func (w *RetrieveWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// writeResponse verarbeitet Embed-Responses und konvertiert sie ins OpenAI-Format
func (w *EmbedWriter) writeResponse(data []byte) (int, error) {
	var embedResponse api.EmbedResponse
	err := json.Unmarshal(data, &embedResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToEmbeddingList(w.model, embedResponse, w.encodingFormat))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer EmbedWriter
func (w *EmbedWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// ListMiddleware erstellt Middleware fuer Model-Listen-Endpoint
func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		w := &ListWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w

		c.Next()
	}
}

// RetrieveMiddleware erstellt Middleware fuer einzelnen Model-Abruf
func RetrieveMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.ShowRequest{Name: c.Param("model")}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &RetrieveWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			model:      c.Param("model"),
		}

		c.Writer = w

		c.Next()
	}
}

// EmbeddingsMiddleware erstellt Middleware fuer Embeddings-Endpoint
func EmbeddingsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.EmbedRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Validate encoding_format parameter
		if req.EncodingFormat != "" {
			if !strings.EqualFold(req.EncodingFormat, "float") && !strings.EqualFold(req.EncodingFormat, "base64") {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("Invalid value for 'encoding_format' = %s. Supported values: ['float', 'base64'].", req.EncodingFormat)))
				return
			}
		}

		if req.Input == "" {
			req.Input = []string{""}
		}

		if req.Input == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		if v, ok := req.Input.([]any); ok && len(v) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.EmbedRequest{Model: req.Model, Input: req.Input, Dimensions: req.Dimensions}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &EmbedWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			model:          req.Model,
			encodingFormat: req.EncodingFormat,
		}

		c.Writer = w

		c.Next()
	}
}
