//go:build windows || darwin

// Modul: server.go
// Beschreibung: Kernfunktionalitaet des Ollama Server-Managers
// Hauptfunktionen:
//   - Server: Verwaltete Ollama-Server-Prozess-Struktur
//   - New(): Erstellt einen neuen Server-Manager
//   - Run(): Hauptschleife fuer den Server-Prozess

package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/app/store"
)

// restartDelay ist die Wartezeit vor einem Neustart des Servers
const restartDelay = time.Second

// Server ist ein verwalteter Ollama-Server-Prozess
type Server struct {
	store *store.Store
	bin   string // Aufgeloester Pfad zu `ollama`
	log   io.WriteCloser
	dev   bool // true wenn mit dev-Flag gestartet
}

// New erstellt einen neuen Server-Manager
func New(s *store.Store, devMode bool) *Server {
	p := resolvePath("ollama")
	return &Server{store: s, bin: p, dev: devMode}
}

// Run startet die Hauptschleife des Server-Prozesses.
// Der Server wird bei Fehlern automatisch neu gestartet.
func (s *Server) Run(ctx context.Context) error {
	l, err := openRotatingLog()
	if err != nil {
		return err
	}
	s.log = l
	defer s.log.Close()

	if err := cleanup(); err != nil {
		slog.Warn("failed to cleanup previous ollama process", "err", err)
	}

	reaped := false
	for ctx.Err() == nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(restartDelay):
		}

		cmd, err := s.cmd(ctx)
		if err != nil {
			return err
		}

		if err := cmd.Start(); err != nil {
			return err
		}

		err = os.WriteFile(pidFile, []byte(strconv.Itoa(cmd.Process.Pid)), 0o644)
		if err != nil {
			slog.Warn("failed to write pid file", "file", pidFile, "err", err)
		}

		if err = cmd.Wait(); err != nil && !errors.Is(err, context.Canceled) {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 && !s.dev && !reaped {
				reaped = true
				// Moeglicher Port-Konflikt, versuche existierende Ollama-Prozesse zu beenden
				if err := reapServers(); err != nil {
					slog.Warn("failed to stop existing ollama server", "err", err)
				} else {
					slog.Debug("conflicting server stopped, waiting for port to be released")
					continue
				}
			}
			slog.Error("ollama exited", "err", err)
		}
	}
	return ctx.Err()
}

// cmd erstellt den exec.Cmd fuer den Ollama-Server mit
// den konfigurierten Umgebungsvariablen aus den Einstellungen.
func (s *Server) cmd(ctx context.Context) (*exec.Cmd, error) {
	settings, err := s.store.Settings()
	if err != nil {
		return nil, err
	}

	cmd := commandContext(ctx, s.bin, "serve")
	cmd.Stdout, cmd.Stderr = s.log, s.log

	// Umgebungsvariablen kopieren und mit Benutzereinstellungen zusammenfuehren
	env := map[string]string{}
	for _, kv := range os.Environ() {
		s := strings.SplitN(kv, "=", 2)
		env[s[0]] = s[1]
	}
	if settings.Expose {
		env["OLLAMA_HOST"] = "0.0.0.0"
	}
	if settings.Browser {
		env["OLLAMA_ORIGINS"] = "*"
	}
	if settings.Models != "" {
		if _, err := os.Stat(settings.Models); err == nil {
			env["OLLAMA_MODELS"] = settings.Models
		} else {
			slog.Warn("models path not accessible, using default", "path", settings.Models, "err", err)
		}
	}
	if settings.ContextLength > 0 {
		env["OLLAMA_CONTEXT_LENGTH"] = strconv.Itoa(settings.ContextLength)
	}
	cmd.Env = []string{}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return stop(cmd.Process)
	}

	return cmd, nil
}
