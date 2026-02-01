// Package server - Chat Handler fuer Chat-Completions
// Beinhaltet: ChatHandler mit Streaming, Tool-Calls, Thinking Support
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
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

// runChatCompletion fuehrt die Chat-Completion aus
func (s *Server) runChatCompletion(c *gin.Context, ch chan any, req api.ChatRequest, r llm.LlamaServer, m *Model, opts *api.Options, prompt string, images []llm.ImageData, msgs []api.Message, processedTools []api.Tool, checkpointStart, checkpointLoaded time.Time, builtinParser parsers.Parser, thinkingState *thinking.Parser, toolParser *tools.Parser, truncate bool) {
	defer close(ch)

	type structuredOutputsState int
	const (
		structuredOutputsState_None structuredOutputsState = iota
		structuredOutputsState_ReadyToApply
		structuredOutputsState_Applying
	)

	structuredOutputsState := structuredOutputsState_None

	for {
		var tb strings.Builder

		currentFormat := req.Format
		if req.Format != nil && structuredOutputsState == structuredOutputsState_None && ((builtinParser != nil || thinkingState != nil) && slices.Contains(m.Capabilities(), model.CapabilityThinking)) {
			currentFormat = nil
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		err := r.Completion(ctx, llm.CompletionRequest{
			Prompt:      prompt,
			Images:      images,
			Format:      currentFormat,
			Options:     opts,
			Shift:       req.Shift == nil || *req.Shift,
			Truncate:    truncate,
			Logprobs:    req.Logprobs,
			TopLogprobs: req.TopLogprobs,
		}, func(cr llm.CompletionResponse) {
			res := api.ChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Message:   api.Message{Role: "assistant", Content: cr.Content},
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
				Logprobs: toAPILogprobs(cr.Logprobs),
			}

			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
			}

			if builtinParser != nil {
				slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser input", "parser", m.Config.Parser, "content", cr.Content)

				content, thinking, toolCalls, err := builtinParser.Add(cr.Content, cr.Done)
				if err != nil {
					ch <- gin.H{"error": err.Error()}
					return
				}

				res.Message.Content = content
				res.Message.Thinking = thinking
				for i := range toolCalls {
					toolCalls[i].ID = toolCallId()
				}
				res.Message.ToolCalls = toolCalls

				tb.WriteString(thinking)
				if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && res.Message.Content != "" {
					structuredOutputsState = structuredOutputsState_ReadyToApply
					cancel()
					return
				}

				if res.Message.Content != "" || res.Message.Thinking != "" || len(res.Message.ToolCalls) > 0 || cr.Done || len(res.Logprobs) > 0 {
					slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser output", "parser", m.Config.Parser, "content", content, "thinking", thinking, "toolCalls", toolCalls, "done", cr.Done)
					ch <- res
				} else {
					slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser empty output", "parser", m.Config.Parser)
				}
				return
			}

			if thinkingState != nil {
				thinkingContent, remainingContent := thinkingState.AddContent(res.Message.Content)
				if thinkingContent == "" && remainingContent == "" && !cr.Done {
					return
				}
				res.Message.Thinking = thinkingContent
				tb.WriteString(thinkingContent)
				if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && remainingContent != "" {
					structuredOutputsState = structuredOutputsState_ReadyToApply
					res.Message.Content = ""
					ch <- res
					cancel()
					return
				}
				res.Message.Content = remainingContent
			}

			if len(req.Tools) > 0 {
				toolCalls, content := toolParser.Add(res.Message.Content)
				if len(content) > 0 {
					res.Message.Content = content
				} else if len(toolCalls) > 0 {
					for i := range toolCalls {
						toolCalls[i].ID = toolCallId()
					}
					res.Message.ToolCalls = toolCalls
					res.Message.Content = ""
				} else if res.Message.Thinking != "" {
					// fall through
				} else {
					if len(res.Logprobs) > 0 && !cr.Done {
						logprobRes := res
						logprobRes.Message.Content = ""
						logprobRes.Message.ToolCalls = nil
						ch <- logprobRes
					}

					if cr.Done {
						res.Message.Content = toolParser.Content()
						ch <- res
					}
					return
				}
			}

			ch <- res
		})
		if err != nil {
			if structuredOutputsState == structuredOutputsState_ReadyToApply && strings.Contains(err.Error(), "context canceled") && c.Request.Context().Err() == nil {
				// ignore
			} else {
				var serr api.StatusError
				if errors.As(err, &serr) {
					ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
				} else {
					ch <- gin.H{"error": err.Error()}
				}
				return
			}
		}

		if structuredOutputsState == structuredOutputsState_ReadyToApply {
			structuredOutputsState = structuredOutputsState_Applying
			msg := api.Message{
				Role:     "assistant",
				Thinking: tb.String(),
			}

			msgs = append(msgs, msg)
			var err error
			prompt, _, err = chatPrompt(c.Request.Context(), m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate)
			if err != nil {
				slog.Error("chat prompt error applying structured outputs", "error", err)
				ch <- gin.H{"error": err.Error()}
				return
			}
			if shouldUseHarmony(m) || (builtinParser != nil && m.Config.Parser == "harmony") {
				prompt += "<|end|><|start|>assistant<|channel|>final<|message|>"
			}
			continue
		}

		break
	}
}

// collectChatResponse sammelt nicht-streaming Chat Response
func (s *Server) collectChatResponse(c *gin.Context, ch chan any, req api.ChatRequest) {
	var resp api.ChatResponse
	var toolCalls []api.ToolCall
	var allLogprobs []api.Logprob
	var sbThinking strings.Builder
	var sbContent strings.Builder
	for rr := range ch {
		switch t := rr.(type) {
		case api.ChatResponse:
			sbThinking.WriteString(t.Message.Thinking)
			sbContent.WriteString(t.Message.Content)
			resp = t
			if len(req.Tools) > 0 {
				toolCalls = append(toolCalls, t.Message.ToolCalls...)
			}
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

	resp.Message.Content = sbContent.String()
	resp.Message.Thinking = sbThinking.String()
	resp.Logprobs = allLogprobs

	if len(toolCalls) > 0 {
		resp.Message.ToolCalls = toolCalls
	}

	c.JSON(http.StatusOK, resp)
}

// handleRemoteChat behandelt Remote-Model Chat-Anfragen
func (s *Server) handleRemoteChat(c *gin.Context, req api.ChatRequest, m *Model) {
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
	if req.Options == nil {
		req.Options = map[string]any{}
	}

	var msgs []api.Message
	if len(req.Messages) > 0 {
		msgs = append(m.Messages, req.Messages...)
		if req.Messages[0].Role != "system" && m.System != "" {
			msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
		}
	}

	msgs = filterThinkTags(msgs, m)
	req.Messages = msgs

	for k, v := range m.Options {
		if _, ok := req.Options[k]; !ok {
			req.Options[k] = v
		}
	}

	contentType := "application/x-ndjson"
	if req.Stream != nil && !*req.Stream {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	fn := func(resp api.ChatResponse) error {
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
	err = client.Chat(c, &req, fn)
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
