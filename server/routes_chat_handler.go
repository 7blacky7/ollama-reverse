// Package server - Chat Handler fuer /api/chat Endpoint
// Beinhaltet: ChatHandler - Haupteinstiegspunkt fuer Chat-Completions
// Unterstuetzt Streaming, Tool-Calls und Thinking
package server

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// ChatHandler verarbeitet /api/chat Anfragen
func (s *Server) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	name, err := getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(req.Model)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		s.handleRemoteChat(c, req, m)
		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if len(req.Tools) > 0 {
		caps = append(caps, model.CapabilityTools)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			if _, ok := c.Get("relax_thinking"); ok {
				slog.Warn("model does not support thinking, relaxing thinking to nil", "model", req.Model)
				req.Think = nil
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
				return
			}
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support chat", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	msgs := append(m.Messages, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)

	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	var builtinParser parsers.Parser
	processedTools := req.Tools

	if m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			processedTools = builtinParser.Init(req.Tools, lastMessage, req.Think)
		}
	}

	truncate := req.Truncate == nil || *req.Truncate
	prompt, images, err := chatPrompt(c.Request.Context(), m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate)
	if err != nil {
		slog.Error("chat prompt error", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(images),
			},
		})
		return
	}

	// Validate Think value
	if req.Think != nil && req.Think.IsString() && m.Config.Parser != "harmony" {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("think value %q is not supported for this model", req.Think.String())})
		return
	}

	var thinkingState *thinking.Parser
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
		thinkingState = &thinking.Parser{
			OpeningTag: openingTag,
			ClosingTag: closingTag,
		}

		if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
			thinkingState.AddContent(openingTag)
		}
	}

	var toolParser *tools.Parser
	if len(req.Tools) > 0 && (builtinParser == nil || !builtinParser.HasToolSupport()) {
		toolParser = tools.NewParser(m.Template.Template, req.Tools)
	}

	ch := make(chan any)
	go s.runChatCompletion(c, ch, req, r, m, opts, prompt, images, msgs, processedTools, checkpointStart, checkpointLoaded, builtinParser, thinkingState, toolParser, truncate)

	if req.Stream != nil && !*req.Stream {
		s.collectChatResponse(c, ch, req)
		return
	}

	streamResponse(c, ch)
}
