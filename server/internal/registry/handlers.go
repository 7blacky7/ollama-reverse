// Package registry - Handler-Implementierungen
// Dieses Modul enthaelt die HTTP-Handler fuer Delete- und Pull-Operationen
// sowie die Fortschritts-Streaming-Logik.
package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/internal/backoff"
)

type progressUpdateJSON struct {
	Error     string      `json:"error,omitempty,omitzero"`
	Status    string      `json:"status,omitempty,omitzero"`
	Digest    blob.Digest `json:"digest,omitempty,omitzero"`
	Total     int64       `json:"total,omitempty,omitzero"`
	Completed int64       `json:"completed,omitempty,omitzero"`
}

func (s *Local) handleDelete(_ http.ResponseWriter, r *http.Request) error {
	if r.Method != "DELETE" {
		return errMethodNotAllowed
	}
	p, err := decodeUserJSON[*params](r.Body)
	if err != nil {
		return err
	}
	ok, err := s.Client.Unlink(p.model())
	if err != nil {
		return err
	}
	if !ok {
		return errModelNotFound
	}
	if s.Prune != nil {
		return s.Prune()
	}
	return nil
}

func (s *Local) handlePull(w http.ResponseWriter, r *http.Request) error {
	if r.Method != "POST" {
		return errMethodNotAllowed
	}

	p, err := decodeUserJSON[*params](r.Body)
	if err != nil {
		return err
	}

	enc := json.NewEncoder(w)
	if !p.stream() {
		if err := s.Client.Pull(r.Context(), p.model()); err != nil {
			if errors.Is(err, ollama.ErrModelNotFound) {
				return errModelNotFound
			}
			return err
		}
		enc.Encode(progressUpdateJSON{Status: "success"})
		return nil
	}

	var mu sync.Mutex
	var progress []progressUpdateJSON
	flushProgress := func() {
		mu.Lock()
		progress := slices.Clone(progress) // make a copy and release lock before encoding to the wire
		mu.Unlock()
		for _, p := range progress {
			enc.Encode(p)
		}
		fl, _ := w.(http.Flusher)
		if fl != nil {
			fl.Flush()
		}
	}

	t := time.NewTicker(1<<63 - 1) // "unstarted" timer
	start := sync.OnceFunc(func() {
		flushProgress() // flush initial state
		t.Reset(100 * time.Millisecond)
	})
	ctx := ollama.WithTrace(r.Context(), &ollama.Trace{
		Update: func(l *ollama.Layer, n int64, err error) {
			if err != nil && !errors.Is(err, ollama.ErrCached) {
				s.Logger.Error("pulling", "model", p.model(), "error", err)
				return
			}

			func() {
				mu.Lock()
				defer mu.Unlock()
				for i, p := range progress {
					if p.Digest == l.Digest {
						progress[i].Completed = n
						return
					}
				}
				progress = append(progress, progressUpdateJSON{
					Digest: l.Digest,
					Total:  l.Size,
				})
			}()

			// Block flushing progress updates until every
			// layer is accounted for. Clients depend on a
			// complete model size to calculate progress
			// correctly; if they use an incomplete total,
			// progress indicators would erratically jump
			// as new layers are registered.
			start()
		},
	})

	done := make(chan error, 1)
	go func() (err error) {
		defer func() { done <- err }()
		for _, err := range backoff.Loop(ctx, 3*time.Second) {
			if err != nil {
				return err
			}
			err := s.Client.Pull(ctx, p.model())
			if canRetry(err) {
				continue
			}
			return err
		}
		return nil
	}()

	enc.Encode(progressUpdateJSON{Status: "pulling manifest"})
	for {
		select {
		case <-t.C:
			flushProgress()
		case err := <-done:
			flushProgress()
			if err != nil {
				if errors.Is(err, ollama.ErrModelNotFound) {
					return &serverError{
						Status:  404,
						Code:    "not_found",
						Message: fmt.Sprintf("model %q not found", p.model()),
					}
				} else {
					return err
				}
			}

			// Emulate old client pull progress (for now):
			enc.Encode(progressUpdateJSON{Status: "verifying sha256 digest"})
			enc.Encode(progressUpdateJSON{Status: "writing manifest"})
			enc.Encode(progressUpdateJSON{Status: "success"})
			return nil
		}
	}
}
