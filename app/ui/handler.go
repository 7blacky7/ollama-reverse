//go:build windows || darwin

// handler.go - HTTP-Handler und Routing
// Enthält: Handler(), ollamaProxy(), handleError(), errHandlerFunc

package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"sync"
	"time"

	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/envconfig"
)

// errHandlerFunc ist ein Handler mit Error-Rückgabe
type errHandlerFunc func(http.ResponseWriter, *http.Request) error

// ollamaProxy erstellt einen Reverse-Proxy zum Ollama-Server
func (s *Server) ollamaProxy() http.Handler {
	var (
		proxy   http.Handler
		proxyMu sync.Mutex
	)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyMu.Lock()
		p := proxy
		proxyMu.Unlock()

		if p == nil {
			proxyMu.Lock()
			if proxy == nil {
				var err error
				for i := range 2 {
					if i > 0 {
						s.log().Warn("ollama server not ready, retrying", "attempt", i+1)
						time.Sleep(1 * time.Second)
					}

					err = WaitForServer(context.Background(), 10*time.Second)
					if err == nil {
						break
					}
				}

				if err != nil {
					proxyMu.Unlock()
					s.log().Error("ollama server not ready after retries", "error", err)
					http.Error(w, "Ollama server is not ready", http.StatusServiceUnavailable)
					return
				}

				target := envconfig.Host()
				s.log().Info("configuring ollama proxy", "target", target.String())

				newProxy := httputil.NewSingleHostReverseProxy(target)

				originalDirector := newProxy.Director
				newProxy.Director = func(req *http.Request) {
					originalDirector(req)
					req.Host = target.Host
					s.log().Debug("proxying request", "method", req.Method, "path", req.URL.Path, "target", target.Host)
				}

				newProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
					s.log().Error("proxy error", "error", err, "path", r.URL.Path, "target", target.String())
					http.Error(w, "proxy error: "+err.Error(), http.StatusBadGateway)
				}

				proxy = newProxy
				p = newProxy
			} else {
				p = proxy
			}
			proxyMu.Unlock()
		}

		p.ServeHTTP(w, r)
	})
}

// Handler erstellt den HTTP-Router mit allen Endpoints
func (s *Server) Handler() http.Handler {
	handle := func(f errHandlerFunc) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// CORS-Header für Entwicklung
			if CORS() {
				w.Header().Set("Access-Control-Allow-Origin", "*")
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
				w.Header().Set("Access-Control-Allow-Credentials", "true")

				if r.Method == "OPTIONS" {
					w.WriteHeader(http.StatusOK)
					return
				}
			}

			// Token-Prüfung (außer im Dev-Modus)
			if !s.Dev {
				cookie, err := r.Cookie("token")
				if err != nil {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}

				if cookie.Value != s.Token {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}
			}

			sw := &statusRecorder{ResponseWriter: w}

			log := s.log()
			level := slog.LevelInfo
			start := time.Now()
			requestID := fmt.Sprintf("%d", time.Now().UnixNano())

			defer func() {
				p := recover()
				if p != nil {
					log = log.With("panic", p, "request_id", requestID)
					level = slog.LevelError

					if !sw.Written() {
						s.handleError(sw, fmt.Errorf("internal server error"))
					}
				}

				log.Log(r.Context(), level, "site.serveHTTP",
					"http.method", r.Method,
					"http.path", r.URL.Path,
					"http.pattern", r.Pattern,
					"http.status", sw.Status(),
					"http.d", time.Since(start),
					"request_id", requestID,
					"version", version.Version,
				)

				if p != nil {
					panic(p)
				}
			}()

			w.Header().Set("X-Frame-Options", "DENY")
			w.Header().Set("X-Version", version.Version)
			w.Header().Set("X-Request-ID", requestID)

			ctx := r.Context()
			if err := f(sw, r); err != nil {
				if ctx.Err() != nil {
					return
				}
				level = slog.LevelError
				log = log.With("error", err)
				s.handleError(sw, err)
			}
		})
	}

	mux := http.NewServeMux()

	// CORS Preflight
	mux.Handle("OPTIONS /", handle(func(w http.ResponseWriter, r *http.Request) error {
		return nil
	}))

	// API-Routen
	mux.Handle("GET /api/v1/chats", handle(s.listChats))
	mux.Handle("GET /api/v1/chat/{id}", handle(s.getChat))
	mux.Handle("POST /api/v1/chat/{id}", handle(s.chat))
	mux.Handle("DELETE /api/v1/chat/{id}", handle(s.deleteChat))
	mux.Handle("POST /api/v1/create-chat", handle(s.createChat))
	mux.Handle("PUT /api/v1/chat/{id}/rename", handle(s.renameChat))

	mux.Handle("GET /api/v1/inference-compute", handle(s.getInferenceCompute))
	mux.Handle("POST /api/v1/model/upstream", handle(s.modelUpstream))
	mux.Handle("GET /api/v1/settings", handle(s.getSettings))
	mux.Handle("POST /api/v1/settings", handle(s.settings))

	// Ollama-Proxy-Endpoints
	ollamaProxy := s.ollamaProxy()
	mux.Handle("GET /api/tags", ollamaProxy)
	mux.Handle("POST /api/show", ollamaProxy)
	mux.Handle("GET /api/version", ollamaProxy)
	mux.Handle("HEAD /api/version", ollamaProxy)
	mux.Handle("POST /api/me", ollamaProxy)
	mux.Handle("POST /api/signout", ollamaProxy)

	// React-App (Catch-All)
	mux.Handle("GET /", s.appHandler())
	mux.Handle("PUT /", s.appHandler())
	mux.Handle("POST /", s.appHandler())
	mux.Handle("PATCH /", s.appHandler())
	mux.Handle("DELETE /", s.appHandler())

	return mux
}

// handleError rendert Fehlerantworten
func (s *Server) handleError(w http.ResponseWriter, e error) {
	if CORS() {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(map[string]string{"error": e.Error()})
}
