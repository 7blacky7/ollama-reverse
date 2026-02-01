//go:build windows || darwin

// settings.go - Settings und Compute Handler
// Enthält: getSettings, settings, getInferenceCompute, modelUpstream

package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/ui/responses"
	"github.com/ollama/ollama/envconfig"
)

// getSettings holt die aktuellen Einstellungen
func (s *Server) getSettings(w http.ResponseWriter, r *http.Request) error {
	settings, err := s.Store.Settings()
	if err != nil {
		return fmt.Errorf("failed to load settings: %w", err)
	}

	if settings.Models == "" {
		settings.Models = envconfig.Models()
	}

	if settings.ContextLength == 0 {
		settings.ContextLength = 4096
	}

	settings.Agent = s.Agent
	settings.Tools = s.Tools
	settings.WorkingDir = s.WorkingDir

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(responses.SettingsResponse{
		Settings: settings,
	})
}

// settings speichert neue Einstellungen
func (s *Server) settings(w http.ResponseWriter, r *http.Request) error {
	old, err := s.Store.Settings()
	if err != nil {
		return fmt.Errorf("failed to load settings: %w", err)
	}

	var settings store.Settings
	if err := json.NewDecoder(r.Body).Decode(&settings); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}

	if err := s.Store.SetSettings(settings); err != nil {
		return fmt.Errorf("failed to save settings: %w", err)
	}

	if old.ContextLength != settings.ContextLength ||
		old.Models != settings.Models ||
		old.Expose != settings.Expose {
		s.Restart()
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(responses.SettingsResponse{
		Settings: settings,
	})
}

// getInferenceCompute holt Informationen über verfügbare Compute-Ressourcen
func (s *Server) getInferenceCompute(w http.ResponseWriter, r *http.Request) error {
	ctx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
	defer cancel()
	serverInferenceComputes, err := server.GetInferenceComputer(ctx)
	if err != nil {
		s.log().Error("failed to get inference compute", "error", err)
		return fmt.Errorf("failed to get inference compute: %w", err)
	}

	inferenceComputes := make([]responses.InferenceCompute, len(serverInferenceComputes))
	for i, ic := range serverInferenceComputes {
		inferenceComputes[i] = responses.InferenceCompute{
			Library: ic.Library,
			Variant: ic.Variant,
			Compute: ic.Compute,
			Driver:  ic.Driver,
			Name:    ic.Name,
			VRAM:    ic.VRAM,
		}
	}

	response := responses.InferenceComputeResponse{
		InferenceComputes: inferenceComputes,
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(response)
}

// modelUpstream prüft Upstream-Informationen eines Models
func (s *Server) modelUpstream(w http.ResponseWriter, r *http.Request) error {
	if r.Method != "POST" {
		return fmt.Errorf("method not allowed")
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}

	if req.Model == "" {
		return fmt.Errorf("model is required")
	}

	digest, pushTime, err := s.checkModelUpstream(r.Context(), req.Model, 5*time.Second)
	if err != nil {
		s.log().Warn("failed to check upstream digest", "error", err, "model", req.Model)
		response := responses.ModelUpstreamResponse{
			Error: err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		return json.NewEncoder(w).Encode(response)
	}

	response := responses.ModelUpstreamResponse{
		Digest:   digest,
		PushTime: pushTime,
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(response)
}
