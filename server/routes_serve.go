// routes_serve.go - Server-Start und Lifecycle-Management
// Enthaelt: Serve() - Hauptfunktion zum Starten des HTTP-Servers

package server

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"syscall"

	"golang.org/x/image/webp"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/version"
)

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
