// Package registry - Hilfsfunktionen
// Dieses Modul enthaelt Utility-Funktionen fuer JSON-Dekodierung,
// Parameter-Handling und Retry-Logik.
package registry

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/ollama/ollama/server/internal/client/ollama"
)

type params struct {
	// DeprecatedName is the name of the model to push, pull, or delete,
	// but is deprecated. New clients should use [Model] instead.
	//
	// Use [model()] to get the model name for both old and new API requests.
	DeprecatedName string `json:"name"`

	// Model is the name of the model to push, pull, or delete.
	//
	// Use [model()] to get the model name for both old and new API requests.
	Model string `json:"model"`

	// AllowNonTLS is a flag that indicates a client using HTTP
	// is doing so, deliberately.
	//
	// Deprecated: This field is ignored and only present for this
	// deprecation message. It should be removed in a future release.
	//
	// Users can just use http or https+insecure to show intent to
	// communicate they want to do insecure things, without awkward and
	// confusing flags such as this.
	AllowNonTLS bool `json:"insecure"`

	// Stream, if true, will make the server send progress updates in a
	// streaming of JSON objects. If false, the server will send a single
	// JSON object with the final status as "success", or an error object
	// if an error occurred.
	//
	// Unfortunately, this API was designed to be a bit awkward. Stream is
	// defined to default to true if not present, so we need a way to check
	// if the client decisively set it to false. So, we use a pointer to a
	// bool. Gross.
	//
	// Use [stream()] to get the correct value for this field.
	Stream *bool `json:"stream"`
}

// model returns the model name for both old and new API requests.
func (p params) model() string {
	return cmp.Or(p.Model, p.DeprecatedName)
}

func (p params) stream() bool {
	if p.Stream == nil {
		return true
	}
	return *p.Stream
}

func decodeUserJSON[T any](r io.Reader) (T, error) {
	var v T
	err := json.NewDecoder(r).Decode(&v)
	if err == nil {
		return v, nil
	}
	var zero T

	// Not sure why, but I can't seem to be able to use:
	//
	//   errors.As(err, &json.UnmarshalTypeError{})
	//
	// This is working fine in stdlib, so I'm not sure what rules changed
	// and why this no longer works here. So, we do it the verbose way.
	var a *json.UnmarshalTypeError
	var b *json.SyntaxError
	if errors.As(err, &a) || errors.As(err, &b) {
		err = &serverError{Status: 400, Message: err.Error(), Code: "bad_request"}
	}
	if errors.Is(err, io.EOF) {
		err = &serverError{Status: 400, Message: "empty request body", Code: "bad_request"}
	}
	return zero, err
}

func canRetry(err error) bool {
	if err == nil {
		return false
	}
	var oe *ollama.Error
	if errors.As(err, &oe) {
		return oe.Temporary()
	}
	s := err.Error()
	return cmp.Or(
		errors.Is(err, context.DeadlineExceeded),
		strings.Contains(s, "unreachable"),
		strings.Contains(s, "no route to host"),
		strings.Contains(s, "connection reset by peer"),
	)
}
