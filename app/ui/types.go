//go:build windows || darwin

// types.go - Typdefinitionen und Konstanten für das UI-Package
// Enthält: Event-Typen, Server-Struct, statusRecorder

package ui

import (
	"log/slog"
	"net/http"
	"os"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/envconfig"
)

// CORS-Konfiguration aus Umgebungsvariable
var CORS = envconfig.Bool("OLLAMA_CORS")

// OllamaDotCom gibt die URL für ollama.com zurück
var OllamaDotCom = func() string {
	if url := os.Getenv("OLLAMA_DOT_COM_URL"); url != "" {
		return url
	}
	return "https://ollama.com"
}()

// statusRecorder zeichnet den HTTP-Status auf
type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) Written() bool {
	return r.code != 0
}

func (r *statusRecorder) WriteHeader(code int) {
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Status() int {
	if r.code == 0 {
		return http.StatusOK
	}
	return r.code
}

func (r *statusRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// Event ist ein String-Typ für SSE-Events
// Der Client verwendet diesen Typ im SSE-Event-Listener
type Event string

const (
	EventChat       Event = "chat"
	EventComplete   Event = "complete"
	EventLoading    Event = "loading"
	EventToolResult Event = "tool_result"
	EventThinking   Event = "thinking"
	EventToolCall   Event = "tool_call"
	EventDownload   Event = "download"
)

// Server ist der Haupt-HTTP-Server für die UI
type Server struct {
	Logger       *slog.Logger
	Restart      func()
	Token        string
	Store        *store.Store
	ToolRegistry *tools.Registry
	Tools        bool   // Single-Turn-Tools aktivieren
	WebSearch    bool   // Single-Turn-Browser-Tool aktivieren
	Agent        bool   // Multi-Turn-Tools aktivieren
	WorkingDir   string // Arbeitsverzeichnis für Agent-Operationen
	Dev          bool   // Entwicklungsmodus
}

// log gibt den konfigurierten Logger zurück
func (s *Server) log() *slog.Logger {
	if s.Logger == nil {
		return slog.Default()
	}
	return s.Logger
}

// userAgentTransport fügt User-Agent-Header zu allen Requests hinzu
type userAgentTransport struct {
	base http.RoundTripper
}

func (t *userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r := req.Clone(req.Context())
	r.Header.Set("User-Agent", userAgent())
	return t.base.RoundTrip(r)
}
