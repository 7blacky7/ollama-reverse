// Package server - Haupt-Router und Server-Setup fuer Ollama
// Beinhaltet: Server-Struct, Router-Registrierung, Middleware, Server-Start
package server

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
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

// isLocalIP prueft ob die IP-Adresse zu einem lokalen Interface gehoert
func isLocalIP(ip netip.Addr) bool {
	if interfaces, err := net.Interfaces(); err == nil {
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}

			for _, a := range addrs {
				if parsed, _, err := net.ParseCIDR(a.String()); err == nil {
					if parsed.String() == ip.String() {
						return true
					}
				}
			}
		}
	}

	return false
}

// allowedHost prueft ob der Host erlaubt ist
func allowedHost(host string) bool {
	host = strings.ToLower(host)

	if host == "" || host == "localhost" {
		return true
	}

	if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
		return true
	}

	tlds := []string{
		"localhost",
		"local",
		"internal",
	}

	// Pruefe ob der Host eine lokale TLD hat
	for _, tld := range tlds {
		if strings.HasSuffix(host, "."+tld) {
			return true
		}
	}

	return false
}

// allowedHostsMiddleware blockiert Anfragen von nicht erlaubten Hosts
func allowedHostsMiddleware(addr net.Addr) gin.HandlerFunc {
	return func(c *gin.Context) {
		if addr == nil {
			c.Next()
			return
		}

		if addr, err := netip.ParseAddrPort(addr.String()); err == nil && !addr.Addr().IsLoopback() {
			c.Next()
			return
		}

		host, _, err := net.SplitHostPort(c.Request.Host)
		if err != nil {
			host = c.Request.Host
		}

		if addr, err := netip.ParseAddr(host); err == nil {
			if addr.IsLoopback() || addr.IsPrivate() || addr.IsUnspecified() || isLocalIP(addr) {
				c.Next()
				return
			}
		}

		if allowedHost(host) {
			if c.Request.Method == http.MethodOptions {
				c.AbortWithStatus(http.StatusNoContent)
				return
			}

			c.Next()
			return
		}

		c.AbortWithStatus(http.StatusForbidden)
	}
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

// Serve startet den HTTP-Server und Scheduler
func Serve(ln net.Listener) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("server config", "env", envconfig.Values())

	blobsDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}
	if err := fixBlobs(blobsDir); err != nil {
		return err
	}

	if !envconfig.NoPrune() {
		if _, err := manifest.Manifests(false); err != nil {
			slog.Warn("corrupt manifests detected, skipping prune operation.  Re-pull or delete to clear", "error", err)
		} else {
			// clean up unused layers and manifests
			if err := PruneLayers(); err != nil {
				return err
			}

			manifestsPath, err := manifest.Path()
			if err != nil {
				return err
			}

			if err := manifest.PruneDirectory(manifestsPath); err != nil {
				return err
			}
		}
	}

	s := &Server{addr: ln.Addr()}

	var rc *ollama.Registry
	if useClient2 {
		var err error
		rc, err = ollama.DefaultRegistry()
		if err != nil {
			return err
		}
	}

	h, err := s.GenerateRoutes(rc)
	if err != nil {
		return err
	}

	http.Handle("/", h)

	ctx, done := context.WithCancel(context.Background())
	schedCtx, schedDone := context.WithCancel(ctx)
	sched := InitScheduler(schedCtx)
	s.sched = sched

	slog.Info(fmt.Sprintf("Listening on %s (version %s)", ln.Addr(), version.Version))
	srvr := &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		//
		// TODO(bmizerany): Decide if we want to make this
		// configurable so it is not exposed by default, or allow
		// users to bind it to a different port. This was a quick
		// and easy way to get pprof, but it may not be the best
		// way.
		Handler: nil,
	}

	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		srvr.Close()
		schedDone()
		sched.unloadAllRunners()
		done()
	}()

	s.sched.Run(schedCtx)

	// register the experimental webp decoder
	// so webp images can be used in multimodal inputs
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)

	// At startup we retrieve GPU information so we can get log messages before loading a model
	// This will log warnings to the log in case we have problems with detected GPUs
	gpus := discover.GPUDevices(ctx, nil)
	discover.LogDetails(gpus)

	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory - envconfig.GpuOverhead()
	}
	if totalVRAM < lowVRAMThreshold {
		s.lowVRAM = true
		slog.Info("entering low vram mode", "total vram", format.HumanBytes2(totalVRAM), "threshold", format.HumanBytes2(lowVRAMThreshold))
	}

	err = srvr.Serve(ln)
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !slices.Contains([]error{http.ErrServerClosed}, err) {
		return err
	}
	<-ctx.Done()
	return nil
}
