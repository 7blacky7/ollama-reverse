//go:build windows || darwin

// Package main - Initialisierungsfunktionen fuer Logging und Server.
// Dieses Modul enthaelt Hilfsfunktionen fuer die App-Initialisierung.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui"
)

// setupLogging initialisiert das Logging-System.
func setupLogging(level slog.Level) (io.Writer, error) {
	if _, err := os.Stat(filepath.Dir(appLogPath)); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(appLogPath), 0o755); err != nil {
			return nil, fmt.Errorf("failed to create server log dir: %w", err)
		}
	}

	logFile, err := os.OpenFile(appLogPath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o755)
	if err != nil {
		return nil, fmt.Errorf("failed to create server log: %w", err)
	}

	var writer io.Writer = logFile
	// Detect if we're a GUI app on windows, and if not, send logs to console as well
	if os.Stderr.Fd() != 0 {
		// Console app detected
		writer = io.MultiWriter(os.Stderr, logFile)
	}

	handler := slog.NewTextHandler(writer, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})

	slog.SetDefault(slog.New(handler))
	return writer, nil
}

// checkViteDevServer prueft, ob der Vite-Dev-Server laeuft.
func checkViteDevServer() error {
	var conn net.Conn
	var err error
	for _, addr := range []string{"127.0.0.1:5173", "localhost:5173"} {
		conn, err = net.DialTimeout("tcp", addr, 2*time.Second)
		if err == nil {
			conn.Close()
			return nil
		}
	}
	return fmt.Errorf("vite dev server not running on port 5173")
}

// serverConfig enthaelt die Konfiguration fuer den Server-Start.
type serverConfig struct {
	ctx          context.Context
	cancel       context.CancelFunc
	store        *store.Store
	token        string
	toolRegistry *tools.Registry
	ln           net.Listener
}

// startServers startet den Ollama- und UI-Server.
func startServers(cfg *serverConfig) (*http.Server, *ui.Server, chan error) {
	// octx is the ollama server context that will be used to stop the ollama server
	octx, ocancel := context.WithCancel(cfg.ctx)

	wv.Store = cfg.store
	done := make(chan error, 1)
	osrv := server.New(cfg.store, devMode)
	go func() {
		slog.Info("starting ollama server")
		done <- osrv.Run(octx)
	}()

	uiServer := &ui.Server{
		Token: cfg.token,
		Restart: func() {
			ocancel()
			<-done
			octx, ocancel = context.WithCancel(cfg.ctx)
			go func() {
				done <- osrv.Run(octx)
			}()
		},
		Store:        cfg.store,
		ToolRegistry: cfg.toolRegistry,
		Dev:          devMode,
		Logger:       slog.Default(),
	}

	srv := &http.Server{
		Handler: uiServer.Handler(),
	}

	// Start the UI server
	slog.Info("starting ui server", "port", cfg.ln.Addr().(*net.TCPAddr).Port)
	go func() {
		slog.Debug("starting ui server on port", "port", cfg.ln.Addr().(*net.TCPAddr).Port)
		err := srv.Serve(cfg.ln)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Warn("desktop server", "error", err)
		}
		slog.Debug("background desktop server done")
	}()

	return srv, uiServer, done
}
