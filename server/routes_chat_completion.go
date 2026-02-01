// Package server - Chat Completion Logik
// Beinhaltet: runChatCompletion, collectChatResponse
// Verantwortlich fuer die Ausfuehrung und Sammlung von Chat-Responses
package server

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/model"
)

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
