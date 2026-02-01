// Package server - Verschiedene Handler und Hilfsfunktionen
// Beinhaltet: Blob-Handler, Auth-Handler, PS-Handler, Stream-Funktionen
package server

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"slices"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/manifest"
)

// HeadBlobHandler prueft ob ein Blob existiert
func (s *Server) HeadBlobHandler(c *gin.Context) {
	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if _, err := os.Stat(path); err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("blob %q not found", c.Param("digest"))})
		return
	}

	c.Status(http.StatusOK)
}

// CreateBlobHandler erstellt einen neuen Blob
func (s *Server) CreateBlobHandler(c *gin.Context) {
	if ib, ok := intermediateBlobs[c.Param("digest")]; ok {
		p, err := manifest.BlobsPath(ib)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, c.Param("digest"))
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		} else {
			c.Status(http.StatusOK)
			return
		}
	}

	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	default:
		c.Status(http.StatusOK)
		return
	}

	layer, err := manifest.NewLayer(c.Request.Body, "")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if layer.Digest != c.Param("digest") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("digest mismatch, expected %q, got %q", c.Param("digest"), layer.Digest)})
		return
	}

	c.Status(http.StatusCreated)
}

// WhoamiHandler gibt den aktuellen Benutzer zurueck
func (s *Server) WhoamiHandler(c *gin.Context) {
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	user, err := client.Whoami(c)
	if err != nil {
		slog.Error(err.Error())
	}

	// user isn't signed in
	if user != nil && user.Name == "" {
		sURL, sErr := signinURL()
		if sErr != nil {
			slog.Error(sErr.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
			return
		}

		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": sURL})
		return
	}

	c.JSON(http.StatusOK, user)
}

// SignoutHandler meldet den Benutzer ab
func (s *Server) SignoutHandler(c *gin.Context) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		slog.Error("couldn't get public key", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))

	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	err = client.Disconnect(c, encKey)
	if err != nil {
		var authError api.AuthorizationError
		if errors.As(err, &authError) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "you are not currently signed in"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	c.JSON(http.StatusOK, nil)
}

// PsHandler gibt die laufenden Modelle zurueck
func (s *Server) PsHandler(c *gin.Context) {
	models := []api.ProcessModelResponse{}

	for _, v := range s.sched.loaded {
		model := v.model
		modelDetails := api.ModelDetails{
			Format:            model.Config.ModelFormat,
			Family:            model.Config.ModelFamily,
			Families:          model.Config.ModelFamilies,
			ParameterSize:     model.Config.ModelType,
			QuantizationLevel: model.Config.FileType,
		}

		mr := api.ProcessModelResponse{
			Model:     model.ShortName,
			Name:      model.ShortName,
			Size:      int64(v.totalSize),
			SizeVRAM:  int64(v.vramSize),
			Digest:    model.Digest,
			Details:   modelDetails,
			ExpiresAt: v.expiresAt,
		}
		if v.Options != nil {
			mr.ContextLength = v.Options.NumCtx
		}
		var epoch time.Time
		if v.expiresAt == epoch {
			mr.ExpiresAt = time.Now().Add(v.sessionDuration)
		}

		models = append(models, mr)
	}

	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})

	c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}

// waitForStream wartet auf nicht-streaming Response
func waitForStream(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/json")
	var latest api.ProgressResponse
	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			latest = r
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			errorMsg, ok := r["error"].(string)
			if !ok {
				errorMsg = "unknown error"
			}
			c.JSON(status, gin.H{"error": errorMsg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
			return
		}
	}

	c.JSON(http.StatusOK, latest)
}

// streamResponse streamt ndjson Responses
func streamResponse(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		if h, ok := val.(gin.H); ok {
			if e, ok := h["error"].(string); ok {
				status, ok := h["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				if !c.Writer.Written() {
					c.Header("Content-Type", "application/json")
					c.JSON(status, gin.H{"error": e})
				} else {
					if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
						slog.Error("streamResponse failed to encode json error", "error", err)
					}
				}

				return false
			}
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}

		return true
	})
}
