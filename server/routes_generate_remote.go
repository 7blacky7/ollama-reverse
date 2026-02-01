// Package server - Remote Generate Handler
// Beinhaltet: handleRemoteGenerate fuer Remote-Model Anfragen
// Delegiert Generate-Requests an entfernte Ollama-Server
package server

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"slices"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

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
