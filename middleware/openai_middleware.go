// openai_middleware.go - Chat und Completions Middlewares
// Verarbeitet Chat-Completions und Standard-Completions Anfragen
package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/openai"
)

// CompletionsMiddleware erstellt Middleware fuer Completions-Endpoint
func CompletionsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.CompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		genReq, err := openai.FromCompleteRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &CompleteWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("cmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w
		c.Next()
	}
}

// ChatMiddleware erstellt Middleware fuer Chat-Completions-Endpoint
func ChatMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ChatCompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer

		chatReq, err := openai.FromChatRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ChatWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("chatcmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w

		c.Next()
	}
}
