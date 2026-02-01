// Package server - Generate Completion Logik
// Beinhaltet: runGenerateCompletion, collectGenerateResponse
// Verantwortlich fuer die Ausfuehrung und Sammlung von Generate-Responses
package server

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
)

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
