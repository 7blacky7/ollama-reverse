// Package server - Image Generate Handler
// Beinhaltet: handleImageGenerate fuer Bild-Generierung
// Verarbeitet Anfragen an Modelle mit Image-Capability
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

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
