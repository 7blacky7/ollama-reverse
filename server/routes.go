// routes.go - Server-Struct und Router-Konfiguration
// Enthaelt: Server struct, GenerateRoutes()

package server

import (
	"log/slog"
	"net"
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/registry"
	"github.com/ollama/ollama/version"
)

// Low VRAM Modus basiert auf Summe des gesamten VRAM (nicht frei)
// und triggert reduzierte Kontextlaenge bei einigen Modellen
var lowVRAMThreshold uint64 = 20 * format.GibiByte

var mode string = gin.DebugMode

// Server verwaltet den HTTP-Server und Scheduler
type Server struct {
	addr    net.Addr
	sched   *Scheduler
	lowVRAM bool
}

func init() {
	switch mode {
	case gin.DebugMode:
	case gin.ReleaseMode:
	case gin.TestMode:
	default:
		mode = gin.DebugMode
	}

	gin.SetMode(mode)

	// Tell renderers to use [img] tags
	renderers.RenderImgTags = true
}

// GenerateRoutes erstellt und konfiguriert den HTTP-Router
func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",

		// OpenAI compatibility headers
		"OpenAI-Beta",
		"x-stainless-arch",
		"x-stainless-async",
		"x-stainless-custom-poll-interval",
		"x-stainless-helper-method",
		"x-stainless-lang",
		"x-stainless-os",
		"x-stainless-package-version",
		"x-stainless-poll-helper",
		"x-stainless-retry-count",
		"x-stainless-runtime",
		"x-stainless-runtime-version",
		"x-stainless-timeout",
	}
	corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true
	r.Use(
		cors.New(corsConfig),
		allowedHostsMiddleware(s.addr),
	)

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })

	// Local model cache management (new implementation is at end of function)
	r.POST("/api/pull", s.PullHandler)
	r.POST("/api/push", s.PushHandler)
	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.DELETE("/api/delete", s.DeleteHandler)

	r.POST("/api/me", s.WhoamiHandler)

	r.POST("/api/signout", s.SignoutHandler)
	// deprecated
	r.DELETE("/api/user/keys/:encodedKey", s.SignoutHandler)

	// Create
	r.POST("/api/create", s.CreateHandler)
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)
	r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)
	r.POST("/api/copy", s.CopyHandler)

	// Inference
	r.GET("/api/ps", s.PsHandler)
	r.POST("/api/generate", s.GenerateHandler)
	r.POST("/api/chat", s.ChatHandler)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)

	// Inference (OpenAI compatibility)
	r.POST("/v1/chat/completions", middleware.ChatMiddleware(), s.ChatHandler)
	r.POST("/v1/completions", middleware.CompletionsMiddleware(), s.GenerateHandler)
	r.POST("/v1/embeddings", middleware.EmbeddingsMiddleware(), s.EmbedHandler)
	r.GET("/v1/models", middleware.ListMiddleware(), s.ListHandler)
	r.GET("/v1/models/:model", middleware.RetrieveMiddleware(), s.ShowHandler)
	r.POST("/v1/responses", middleware.ResponsesMiddleware(), s.ChatHandler)
	// OpenAI-compatible image generation endpoints
	r.POST("/v1/images/generations", middleware.ImageGenerationsMiddleware(), s.GenerateHandler)
	r.POST("/v1/images/edits", middleware.ImageEditsMiddleware(), s.GenerateHandler)

	// Inference (Anthropic compatibility)
	r.POST("/v1/messages", middleware.AnthropicMessagesMiddleware(), s.ChatHandler)

	if rc != nil {
		// wrap old with new
		rs := &registry.Local{
			Client:   rc,
			Logger:   slog.Default(), // TODO(bmizerany): Take a logger, do not use slog.Default()
			Fallback: r,

			Prune: PruneLayers,
		}
		return rs, nil
	}

	return r, nil
}
