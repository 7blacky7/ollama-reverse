// Package server - Generate Handler fuer /api/generate Endpoint
// Beinhaltet: GenerateHandler - Haupteinstiegspunkt fuer Text-Generierung
package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// GenerateHandler verarbeitet /api/generate Anfragen
func (s *Server) GenerateHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.GenerateRequest
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
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	name, err := getExistingName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
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

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		s.handleRemoteGenerate(c, req, m)
		return
	}

	// expire the runner if unload is requested (empty prompt, keep alive is 0)
	if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	// Handle image generation models
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		s.handleImageGenerate(c, req, name.String(), checkpointStart)
		return
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	var builtinParser parsers.Parser
	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	if !req.Raw && m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			builtinParser.Init(nil, nil, req.Think)
		}
	}

	// Validate Think value: string values currently only allowed for harmony/gptoss models
	if req.Think != nil && req.Think.IsString() && m.Config.Parser != "harmony" {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("think value %q is not supported for this model", req.Think.String())})
		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image while more than one image requested"})
		return
	}

	images := make([]llm.ImageData, len(req.Images))
	for i := range req.Images {
		images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
	}

	prompt := req.Prompt
	if !req.Raw {
		prompt, images, err = s.buildGeneratePrompt(c, req, m, r, opts, images)
		if err != nil {
			return
		}
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(images),
			},
		})
		return
	}

	var thinkingState *thinking.Parser
	if builtinParser == nil {
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
	}

	ch := make(chan any)
	go s.runGenerateCompletion(c, ch, req, r, prompt, images, opts, checkpointStart, checkpointLoaded, builtinParser, thinkingState)

	if req.Stream != nil && !*req.Stream {
		s.collectGenerateResponse(c, ch)
		return
	}

	streamResponse(c, ch)
}

// buildGeneratePrompt baut den Prompt fuer Generate-Anfragen
func (s *Server) buildGeneratePrompt(c *gin.Context, req api.GenerateRequest, m *Model, r llm.LlamaServer, opts *api.Options, images []llm.ImageData) (string, []llm.ImageData, error) {
	tmpl := m.Template
	var err error
	if req.Template != "" {
		tmpl, err = template.Parse(req.Template)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return "", nil, err
		}
	}

	var values template.Values
	if req.Suffix != "" {
		values.Prompt = req.Prompt
		values.Suffix = req.Suffix
	} else {
		var msgs []api.Message
		if req.System != "" {
			msgs = append(msgs, api.Message{Role: "system", Content: req.System})
		} else if m.System != "" {
			msgs = append(msgs, api.Message{Role: "system", Content: m.System})
		}

		if req.Context == nil {
			msgs = append(msgs, m.Messages...)
		}

		userMsg := api.Message{Role: "user", Content: req.Prompt}
		for _, i := range images {
			userMsg.Images = append(userMsg.Images, i.Data)
		}
		values.Messages = append(msgs, userMsg)
	}

	values.Think = req.Think != nil && req.Think.Bool()
	values.ThinkLevel = ""
	if req.Think != nil {
		values.ThinkLevel = req.Think.String()
	}
	values.IsThinkSet = req.Think != nil

	var b bytes.Buffer
	if req.Context != nil {
		slog.Warn("the context field is deprecated and will be removed in a future version of Ollama")
		s, err := r.Detokenize(c.Request.Context(), req.Context)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return "", nil, err
		}
		b.WriteString(s)
	}

	// check that we're in the `api/chat`-like flow, and if so, generate the
	// prompt the same way
	if values.Messages != nil && values.Suffix == "" && req.Template == "" {
		prompt, imgs, err := chatPrompt(c.Request.Context(), m, r.Tokenize, opts, values.Messages, []api.Tool{}, req.Think, req.Truncate == nil || *req.Truncate)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return "", nil, err
		}
		if req.Context != nil {
			b.WriteString(prompt)
			prompt = b.String()
		}
		return prompt, imgs, nil
	}

	// legacy flow
	if err := tmpl.Execute(&b, values); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return "", nil, err
	}

	return b.String(), images, nil
}
