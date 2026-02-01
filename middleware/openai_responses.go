// openai_responses.go - Responses API Writer und Middleware
// Implementiert die OpenAI Responses API Kompatibilitaet
package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// ResponsesWriter verarbeitet Responses-API-Anfragen
type ResponsesWriter struct {
	BaseWriter
	converter  *openai.ResponsesStreamConverter
	model      string
	stream     bool
	responseID string
	itemID     string
	request    openai.ResponsesRequest
}

// writeEvent schreibt ein SSE-Event
func (w *ResponsesWriter) writeEvent(eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, d)))
	if err != nil {
		return err
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

// writeResponse verarbeitet Chat-Responses fuer die Responses API
func (w *ResponsesWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	if w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		events := w.converter.Process(chatResponse)
		for _, event := range events {
			if err := w.writeEvent(event.Event, event.Data); err != nil {
				return 0, err
			}
		}
		return len(data), nil
	}

	// Non-streaming response
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

// Write implementiert io.Writer fuer ResponsesWriter
func (w *ResponsesWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}
	return w.writeResponse(data)
}

// ResponsesMiddleware erstellt Middleware fuer Responses-API-Endpoint
func ResponsesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ResponsesRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		chatReq, err := openai.FromResponsesRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Check if client requested streaming (defaults to false)
		streamRequested := req.Stream != nil && *req.Stream

		// Pass streaming preference to the underlying chat request
		chatReq.Stream = &streamRequested

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		responseID := fmt.Sprintf("resp_%d", rand.Intn(999999))
		itemID := fmt.Sprintf("msg_%d", rand.Intn(999999))

		w := &ResponsesWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			converter:  openai.NewResponsesStreamConverter(responseID, itemID, req.Model, req),
			model:      req.Model,
			stream:     streamRequested,
			responseID: responseID,
			itemID:     itemID,
			request:    req,
		}

		// Set headers based on streaming mode
		if streamRequested {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		c.Writer = w
		c.Next()
	}
}
