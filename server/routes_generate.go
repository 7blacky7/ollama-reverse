// Package server - Generate und Chat Handler fuer Text-Generierung
// Beinhaltet: GenerateHandler, ChatHandler, handleImageGenerate
package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

const signinURLStr = "https://ollama.com/connect?name=%s&key=%s"

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

// shouldUseHarmony prueft ob Harmony-Parser verwendet werden soll
func shouldUseHarmony(model *Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, model.Config.ModelFamily) {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

// experimentEnabled prueft ob ein Experiment aktiviert ist
func experimentEnabled(name string) bool {
	return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")

// modelOptions merged Model-Optionen mit Request-Optionen
func modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	opts := api.DefaultOptions()
	if err := opts.FromMap(model.Options); err != nil {
		return api.Options{}, err
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	return opts, nil
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, and consolidated options if successful and error otherwise.
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	m, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(m.ProjectorPaths) > 0 {
		return nil, nil, nil, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	if err := m.CheckCapabilities(caps...); err != nil {
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	opts, err := modelOptions(m, requestOpts)
	if err != nil {
		return nil, nil, nil, err
	}

	// This model is much more capable with a larger context, so set that
	// unless it would penalize performance too much
	if !s.lowVRAM && slices.Contains([]string{
		"gptoss", "gpt-oss",
		"qwen3vl", "qwen3vlmoe",
	}, m.Config.ModelFamily) {
		opts.NumCtx = max(opts.NumCtx, 8192)
	}

	runnerCh, errCh := s.sched.GetRunner(ctx, m, opts, keepAlive)
	var runner *runnerRef
	select {
	case runner = <-runnerCh:
	case err = <-errCh:
		return nil, nil, nil, err
	}

	return runner.llama, m, &opts, nil
}

// signinURL generiert die Sign-In URL
func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf(signinURLStr, url.PathEscape(h), encKey), nil
}

// toolCallId generiert eine zufaellige Tool-Call ID
func toolCallId() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

// handleScheduleError behandelt Scheduler-Fehler
func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.Canceled):
		c.JSON(499, gin.H{"error": "request canceled"})
	case errors.Is(err, ErrMaxQueue):
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

// filterThinkTags filtert Think-Tags aus Nachrichten
func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}

		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				// TODO(drifkin): this is from before we added proper thinking support.
				// However, even if thinking is not enabled (and therefore we shouldn't
				// change the user output), we should probably perform this filtering
				// for all thinking models (not just qwen3 & deepseek-r1) since it tends
				// to save tokens and improve quality.
				thinkingState := &thinking.Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}

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

// runGenerateCompletion fuehrt die Completion aus und sendet an Channel
func (s *Server) runGenerateCompletion(c *gin.Context, ch chan any, req api.GenerateRequest, r llm.LlamaServer, prompt string, images []llm.ImageData, opts *api.Options, checkpointStart, checkpointLoaded time.Time, builtinParser parsers.Parser, thinkingState *thinking.Parser) {
	var sb strings.Builder
	defer close(ch)
	if err := r.Completion(c.Request.Context(), llm.CompletionRequest{
		Prompt:      prompt,
		Images:      images,
		Format:      req.Format,
		Options:     opts,
		Shift:       req.Shift == nil || *req.Shift,
		Truncate:    req.Truncate == nil || *req.Truncate,
		Logprobs:    req.Logprobs,
		TopLogprobs: req.TopLogprobs,
	}, func(cr llm.CompletionResponse) {
		res := api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			Response:  cr.Content,
			Done:      cr.Done,
			Metrics: api.Metrics{
				PromptEvalCount:    cr.PromptEvalCount,
				PromptEvalDuration: cr.PromptEvalDuration,
				EvalCount:          cr.EvalCount,
				EvalDuration:       cr.EvalDuration,
			},
			Logprobs: toAPILogprobs(cr.Logprobs),
		}

		if builtinParser != nil {
			content, thinking, toolCalls, err := builtinParser.Add(cr.Content, cr.Done)
			if err != nil {
				ch <- gin.H{"error": err.Error()}
				return
			}
			res.Response = content
			res.Thinking = thinking
			if cr.Done && len(toolCalls) > 0 {
				res.ToolCalls = toolCalls
			}
		} else if thinkingState != nil {
			thinking, content := thinkingState.AddContent(cr.Content)
			res.Thinking = thinking
			res.Response = content
		}

		if _, err := sb.WriteString(cr.Content); err != nil {
			ch <- gin.H{"error": err.Error()}
		}

		if cr.Done {
			res.DoneReason = cr.DoneReason.String()
			res.TotalDuration = time.Since(checkpointStart)
			res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

			if !req.Raw {
				tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
				if err != nil {
					ch <- gin.H{"error": err.Error()}
					return
				}
				res.Context = tokens
			}
		}

		if builtinParser != nil {
			if res.Response != "" || res.Thinking != "" || res.Done || len(res.ToolCalls) > 0 {
				ch <- res
			}
			return
		}

		ch <- res
	}); err != nil {
		var serr api.StatusError
		if errors.As(err, &serr) {
			ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
		} else {
			ch <- gin.H{"error": err.Error()}
		}
	}
}

// collectGenerateResponse sammelt nicht-streaming Response
func (s *Server) collectGenerateResponse(c *gin.Context, ch chan any) {
	var r api.GenerateResponse
	var allLogprobs []api.Logprob
	var sbThinking strings.Builder
	var sbContent strings.Builder
	for rr := range ch {
		switch t := rr.(type) {
		case api.GenerateResponse:
			sbThinking.WriteString(t.Thinking)
			sbContent.WriteString(t.Response)
			r = t
			if len(t.Logprobs) > 0 {
				allLogprobs = append(allLogprobs, t.Logprobs...)
			}
		case gin.H:
			msg, ok := t["error"].(string)
			if !ok {
				msg = "unexpected error format in response"
			}

			status, ok := t["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}

			c.JSON(status, gin.H{"error": msg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
			return
		}
	}

	r.Thinking = sbThinking.String()
	r.Response = sbContent.String()
	r.Logprobs = allLogprobs

	c.JSON(http.StatusOK, r)
}

// handleRemoteGenerate behandelt Remote-Model Generate-Anfragen
func (s *Server) handleRemoteGenerate(c *gin.Context, req api.GenerateRequest, m *Model) {
	origModel := req.Model

	remoteURL, err := url.Parse(m.Config.RemoteHost)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if !slices.Contains(envconfig.Remotes(), remoteURL.Hostname()) {
		slog.Info("remote model", "remotes", envconfig.Remotes(), "remoteURL", m.Config.RemoteHost, "hostname", remoteURL.Hostname())
		c.JSON(http.StatusBadRequest, gin.H{"error": "this server cannot run this remote model"})
		return
	}

	req.Model = m.Config.RemoteModel

	if req.Template == "" && m.Template.String() != "" {
		req.Template = m.Template.String()
	}

	if req.Options == nil {
		req.Options = map[string]any{}
	}

	for k, v := range m.Options {
		if _, ok := req.Options[k]; !ok {
			req.Options[k] = v
		}
	}

	if req.System == "" && m.System != "" {
		req.System = m.System
	}

	if len(m.Messages) > 0 {
		slog.Warn("embedded messages in the model not supported with '/api/generate'; try '/api/chat' instead")
	}

	contentType := "application/x-ndjson"
	if req.Stream != nil && !*req.Stream {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	fn := func(resp api.GenerateResponse) error {
		resp.Model = origModel
		resp.RemoteModel = m.Config.RemoteModel
		resp.RemoteHost = m.Config.RemoteHost

		data, err := json.Marshal(resp)
		if err != nil {
			return err
		}

		if _, err = c.Writer.Write(append(data, '\n')); err != nil {
			return err
		}
		c.Writer.Flush()
		return nil
	}

	client := api.NewClient(remoteURL, http.DefaultClient)
	err = client.Generate(c, &req, fn)
	if err != nil {
		var authError api.AuthorizationError
		if errors.As(err, &authError) {
			sURL, sErr := signinURL()
			if sErr != nil {
				slog.Error(sErr.Error())
				c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
				return
			}

			c.JSON(authError.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
			return
		}
		var apiError api.StatusError
		if errors.As(err, &apiError) {
			c.JSON(apiError.StatusCode, apiError)
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

// handleImageGenerate handles image generation requests within GenerateHandler.
// This is called when the model has the Image capability.
func (s *Server) handleImageGenerate(c *gin.Context, req api.GenerateRequest, modelName string, checkpointStart time.Time) {
	const maxDimension int32 = 4096
	if req.Width > maxDimension || req.Height > maxDimension {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("width and height must be <= %d", maxDimension)})
		return
	}

	runner, _, _, err := s.scheduleRunner(c.Request.Context(), modelName, []model.Capability{model.CapabilityImage}, nil, req.KeepAlive)
	if err != nil {
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

	isStreaming := req.Stream == nil || *req.Stream

	contentType := "application/x-ndjson"
	if !isStreaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	var seed int64
	if s, ok := req.Options["seed"]; ok {
		switch v := s.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}

	var images []llm.ImageData
	for i, imgData := range req.Images {
		images = append(images, llm.ImageData{ID: i, Data: imgData})
	}

	var streamStarted bool
	var finalResponse api.GenerateResponse

	if err := runner.Completion(c.Request.Context(), llm.CompletionRequest{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  req.Steps,
		Seed:   seed,
		Images: images,
	}, func(cr llm.CompletionResponse) {
		streamStarted = true
		res := api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			Done:      cr.Done,
		}

		if cr.TotalSteps > 0 {
			res.Completed = int64(cr.Step)
			res.Total = int64(cr.TotalSteps)
		}

		if cr.Image != "" {
			res.Image = cr.Image
		}

		if cr.Done {
			res.DoneReason = cr.DoneReason.String()
			res.Metrics.TotalDuration = time.Since(checkpointStart)
			res.Metrics.LoadDuration = checkpointLoaded.Sub(checkpointStart)
		}

		if !isStreaming {
			finalResponse = res
			return
		}

		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
	}); err != nil {
		if !streamStarted {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if !isStreaming {
		c.JSON(http.StatusOK, finalResponse)
	}
}
