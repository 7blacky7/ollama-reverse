// Package registry implementiert einen http.Handler fuer lokale Ollama API
// Model-Management-Anfragen. Dieses Modul enthaelt den Haupt-Handler
// und die HTTP-Routing-Logik.
package registry

import (
	"cmp"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/server/internal/client/ollama"
)

// Local implements an http.Handler for handling local Ollama API model
// management requests, such as pushing, pulling, and deleting models.
//
// It can be arranged for all unknown requests to be passed through to a
// fallback handler, if one is provided.
type Local struct {
	Client *ollama.Registry // required
	Logger *slog.Logger     // required

	// Fallback, if set, is used to handle requests that are not handled by
	// this handler.
	Fallback http.Handler

	// Prune, if set, is called to prune the local disk cache after a model
	// is deleted.
	Prune func() error // optional
}

// serverError is like ollama.Error, but with a Status field for the HTTP
// response code. We want to avoid adding that field to ollama.Error because it
// would always be 0 to clients (we don't want to leak the status code in
// errors), and so it would be confusing to have a field that is always 0.
type serverError struct {
	Status int `json:"-"`

	// TODO(bmizerany): Decide if we want to keep this and maybe
	// bring back later.
	Code string `json:"code"`

	Message string `json:"error"`
}

func (e serverError) Error() string {
	return e.Message
}

// Common API errors
var (
	errMethodNotAllowed = &serverError{405, "method_not_allowed", "method not allowed"}
	errNotFound         = &serverError{404, "not_found", "not found"}
	errModelNotFound    = &serverError{404, "not_found", "model not found"}
	errInternalError    = &serverError{500, "internal_error", "internal server error"}
)

type statusCodeRecorder struct {
	_status int // use status() to get the status code
	http.ResponseWriter
}

func (r *statusCodeRecorder) WriteHeader(status int) {
	if r._status == 0 {
		r._status = status
		r.ResponseWriter.WriteHeader(status)
	}
}

func (r *statusCodeRecorder) Write(b []byte) (int, error) {
	r._status = r.status()
	return r.ResponseWriter.Write(b)
}

var (
	_ http.ResponseWriter = (*statusCodeRecorder)(nil)
	_ http.CloseNotifier  = (*statusCodeRecorder)(nil)
	_ http.Flusher        = (*statusCodeRecorder)(nil)
)

// CloseNotify implements the http.CloseNotifier interface, for Gin. Remove with Gin.
//
// It panics if the underlying ResponseWriter is not a CloseNotifier.
func (r *statusCodeRecorder) CloseNotify() <-chan bool {
	return r.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

// Flush implements the http.Flusher interface, for Gin. Remove with Gin.
//
// It panics if the underlying ResponseWriter is not a Flusher.
func (r *statusCodeRecorder) Flush() {
	r.ResponseWriter.(http.Flusher).Flush()
}

func (r *statusCodeRecorder) status() int {
	return cmp.Or(r._status, 200)
}

func (s *Local) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rec := &statusCodeRecorder{ResponseWriter: w}
	s.serveHTTP(rec, r)
}

func (s *Local) serveHTTP(rec *statusCodeRecorder, r *http.Request) {
	var errattr slog.Attr
	proxied, err := func() (bool, error) {
		switch r.URL.Path {
		case "/api/delete":
			return false, s.handleDelete(rec, r)
		case "/api/pull":
			return false, s.handlePull(rec, r)
		default:
			if s.Fallback != nil {
				s.Fallback.ServeHTTP(rec, r)
				return true, nil
			}
			return false, errNotFound
		}
	}()
	if err != nil {
		// We always log the error, so fill in the error log attribute
		errattr = slog.String("error", err.Error())

		var e *serverError
		switch {
		case errors.As(err, &e):
		case errors.Is(err, ollama.ErrNameInvalid):
			e = &serverError{400, "bad_request", err.Error()}
		default:
			e = errInternalError
		}

		data, err := json.Marshal(e)
		if err != nil {
			// unreachable
			panic(err)
		}
		rec.Header().Set("Content-Type", "application/json")
		rec.WriteHeader(e.Status)
		rec.Write(data)

		// fallthrough to log
	}

	if !proxied {
		// we're only responsible for logging if we handled the request
		var level slog.Level
		if rec.status() >= 500 {
			level = slog.LevelError
		} else if rec.status() >= 400 {
			level = slog.LevelWarn
		}

		s.Logger.LogAttrs(r.Context(), level, "http",
			errattr, // report first in line to make it easy to find

			// TODO(bmizerany): Write a test to ensure that we are logging
			// all of this correctly. That also goes for the level+error
			// logic above.
			slog.Int("status", rec.status()),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int64("content-length", r.ContentLength),
			slog.String("remote", r.RemoteAddr),
			slog.String("proto", r.Proto),
			slog.String("query", r.URL.RawQuery),
		)
	}
}
