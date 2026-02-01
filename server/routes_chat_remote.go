// Package server - Remote Chat Handler
// Beinhaltet: handleRemoteChat fuer Remote-Model Chat-Anfragen
// Delegiert Chat-Requests an entfernte Ollama-Server
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
