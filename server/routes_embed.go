// Package server - Embedding Handler fuer Text-Embeddings
// Beinhaltet: EmbedHandler, EmbeddingsHandler (deprecated), normalize
package server

import (
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// EmbedHandler verarbeitet /api/embed Anfragen (neuer Endpoint)
func (s *Server) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}

	name, err := getExistingName(model.ParseName(req.Model))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()

	embedWithRetry := func(text string) ([]float32, int, error) {
		emb, tokCount, err := r.Embedding(ctx, text)
		if err == nil {
			return emb, tokCount, nil
		}

		var serr api.StatusError
		if !errors.As(err, &serr) || serr.StatusCode != http.StatusBadRequest {
			return nil, 0, err
		}
		if req.Truncate != nil && !*req.Truncate {
			return nil, 0, err
		}

		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return nil, 0, err
		}

		ctxLen := min(opts.NumCtx, int(kvData.ContextLength()))
		if bos := kvData.Uint("tokenizer.ggml.bos_token_id"); len(tokens) > 0 && tokens[0] != int(bos) && kvData.Bool("add_bos_token", true) {
			ctxLen--
		}
		if eos := kvData.Uint("tokenizer.ggml.eos_token_id"); len(tokens) > 0 && tokens[len(tokens)-1] != int(eos) && kvData.Bool("add_eos_token", true) {
			ctxLen--
		}

		if len(tokens) <= ctxLen {
			return nil, 0, fmt.Errorf("input exceeds maximum context length and cannot be truncated further")
		}
		if ctxLen <= 0 {
			return nil, 0, fmt.Errorf("input after truncation exceeds maximum context length")
		}

		truncatedTokens := tokens[:ctxLen]
		truncated, err := r.Detokenize(ctx, truncatedTokens)
		if err != nil {
			return nil, 0, err
		}
		return r.Embedding(ctx, truncated)
	}

	var g errgroup.Group
	embeddings := make([][]float32, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			embedding, tokenCount, err := embedWithRetry(text)
			if err != nil {
				return err
			}
			embedding, err = normalize(embedding)
			if err != nil {
				return err
			}
			if req.Dimensions > 0 && req.Dimensions < len(embedding) {
				embedding, err = normalize(embedding[:req.Dimensions])
				if err != nil {
					return err
				}
			}
			embeddings[i] = embedding
			atomic.AddUint64(&totalTokens, uint64(tokenCount))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{
				"error": strings.TrimSpace(serr.ErrorMessage),
			})
			return
		}

		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": strings.TrimSpace(err.Error()),
		})
		return
	}

	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	}
	c.JSON(http.StatusOK, resp)
}

// normalize normalisiert einen Embedding-Vektor
func normalize(vec []float32) ([]float32, error) {
	var sum float32
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("embedding contains NaN or Inf values")
		}
		sum += v * v
	}

	norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
	for i := range vec {
		vec[i] *= norm
	}
	return vec, nil
}

// EmbeddingsHandler verarbeitet /api/embeddings Anfragen (deprecated)
func (s *Server) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	embedding, _, err := r.Embedding(c.Request.Context(), req.Prompt)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := api.EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}
