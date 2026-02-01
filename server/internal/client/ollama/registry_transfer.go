// Package ollama - Push/Pull Operationen für Model-Transfers
package ollama

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/server/internal/cache/blob"
)

// PushParams enthält optionale Parameter für Push
type PushParams struct {
	From string // optionaler Quellname
}

// Push lädt das Modell mit dem angegebenen Namen in die Remote-Registry hoch
func (r *Registry) Push(ctx context.Context, name string, p *PushParams) error {
	if p == nil {
		p = &PushParams{}
	}

	c, err := r.cache()
	if err != nil {
		return err
	}

	m, err := r.ResolveLocal(cmp.Or(p.From, name))
	if err != nil {
		return err
	}

	// Validiere alle Layer bevor der Upload startet
	for _, l := range m.Layers {
		if l == nil {
			return fmt.Errorf("%w: null layer", ErrManifestInvalid)
		}
		info, err := c.Get(l.Digest)
		if err != nil {
			return fmt.Errorf("error getting %s: %w", l.Digest.Short(), err)
		}
		if info.Size != l.Size {
			return fmt.Errorf("size mismatch for %s: %d != %d", l.Digest.Short(), info.Size, l.Size)
		}
	}

	t := traceFromContext(ctx)

	scheme, n, _, err := r.parseNameExtended(name)
	if err != nil {
		panic(err) // Sollte nie passieren, ResolveLocal hat bereits validiert
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var g errgroup.Group
	g.SetLimit(r.maxStreams())
	for _, l := range m.Layers {
		var progress atomic.Int64
		g.Go(func() (err error) {
			defer func() { t.update(l, progress.Load(), err) }()

			t.update(l, 0, nil)

			startURL := fmt.Sprintf("%s://%s/v2/%s/%s/blobs/uploads/?digest=%s",
				scheme,
				n.Host(),
				n.Namespace(),
				n.Model(),
				l.Digest,
			)
			res, err := r.send(ctx, "POST", startURL, nil)
			if err != nil {
				return err
			}
			res.Body.Close()

			f, err := os.Open(c.GetFile(l.Digest))
			if err != nil {
				return err
			}
			defer f.Close()

			uploadURL := res.Header.Get("Location")
			if uploadURL == "" {
				t.update(l, l.Size, ErrCached)
				return nil
			}

			req, err := r.newRequest(ctx, "PUT", uploadURL, f)
			if err != nil {
				return fmt.Errorf("invalid upload URL returned from registry: %q: %w", uploadURL, err)
			}
			req.ContentLength = l.Size

			res, err = sendRequest(r.client(), req)
			if err == nil {
				res.Body.Close()
			}
			return err
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// Commit des Manifests
	path := fmt.Sprintf("%s://%s/v2/%s/%s/manifests/%s",
		scheme,
		n.Host(),
		n.Namespace(),
		n.Model(),
		n.Tag(),
	)
	res, err := r.send(ctx, "PUT", path, bytes.NewReader(m.Data))
	if err == nil {
		res.Body.Close()
	}
	return err
}

// trackingReader verfolgt gelesene Bytes und ruft update auf
type trackingReader struct {
	r      io.Reader
	update func(n int64, err error)
}

func (r *trackingReader) Read(p []byte) (n int, err error) {
	n, err = r.r.Read(p)
	r.update(int64(n), nil)
	return
}

// Pull lädt das Modell von der Remote-Registry in den Cache
// Große Layer werden in Chunks heruntergeladen
func (r *Registry) Pull(ctx context.Context, name string) error {
	m, err := r.Resolve(ctx, name)
	if err != nil {
		return err
	}

	if len(m.Layers) == 0 {
		return fmt.Errorf("%w: no layers", ErrManifestInvalid)
	}

	c, err := r.cache()
	if err != nil {
		return err
	}

	// Config-Layer für Legacy-Kompatibilität hinzufügen
	layers := m.Layers
	if m.Config != nil && m.Config.Digest.IsValid() {
		layers = append(layers, m.Config)
	}

	// Initiale Trace-Events für Fortschrittsanzeige
	var expected int64
	t := traceFromContext(ctx)
	for _, l := range layers {
		t.update(l, 0, nil)
		expected += l.Size
	}

	var g errgroup.Group
	g.SetLimit(r.maxStreams())

	var completed atomic.Int64
	for _, l := range layers {
		var received atomic.Int64
		update := func(n int64, err error) {
			if n == 0 && err == nil {
				return // Kein Update bei Start ohne Fortschritt
			}
			completed.Add(n)
			t.update(l, received.Add(n), err)
		}

		info, err := c.Get(l.Digest)
		if err == nil && info.Size == l.Size {
			update(l.Size, ErrCached)
			continue
		}

		func() (err error) {
			defer func() {
				if err != nil {
					update(0, err)
				}
			}()

			var wg sync.WaitGroup
			chunked, err := c.Chunked(l.Digest, l.Size)
			if err != nil {
				return err
			}
			defer func() {
				g.Go(func() error {
					wg.Wait()
					chunked.Close()
					return nil
				})
			}()

			for cs, err := range r.chunksums(ctx, name, l) {
				if err != nil {
					update(0, err)
					break
				}

				cacheKey := fmt.Sprintf(
					"v1 pull chunksum %s %s %d-%d",
					l.Digest,
					cs.Digest,
					cs.Chunk.Start,
					cs.Chunk.End,
				)
				cacheKeyDigest := blob.DigestFromBytes(cacheKey)
				_, err := c.Get(cacheKeyDigest)
				if err == nil {
					update(cs.Chunk.Size(), ErrCached)
					continue
				}

				wg.Add(1)
				g.Go(func() (err error) {
					defer func() {
						defer wg.Done()
						if err != nil {
							update(0, err)
						}
					}()

					ctx, cancel := context.WithCancelCause(ctx)
					defer cancel(nil)

					timer := time.AfterFunc(r.readTimeout(), func() {
						cancel(fmt.Errorf("%w: downloading %s %d-%d/%d",
							context.DeadlineExceeded,
							cs.Digest.Short(),
							cs.Chunk.Start,
							cs.Chunk.End,
							l.Size,
						))
					})
					defer timer.Stop()

					req, err := http.NewRequestWithContext(ctx, "GET", cs.URL, nil)
					if err != nil {
						return err
					}
					req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", cs.Chunk.Start, cs.Chunk.End))
					res, err := sendRequest(r.client(), req)
					if err != nil {
						return err
					}
					defer res.Body.Close()

					tr := &trackingReader{
						r: res.Body,
						update: func(n int64, err error) {
							timer.Reset(r.readTimeout())
							update(n, err)
						},
					}
					if err := chunked.Put(cs.Chunk, cs.Digest, tr); err != nil {
						return err
					}

					return blob.PutBytes(c, cacheKeyDigest, cacheKey)
				})
			}

			return nil
		}()
	}
	if err := g.Wait(); err != nil {
		return err
	}
	if recv := completed.Load(); recv != expected {
		return fmt.Errorf("%w: received %d/%d bytes", ErrIncomplete, recv, expected)
	}

	md := blob.DigestFromBytes(m.Data)
	if err := blob.PutBytes(c, md, m.Data); err != nil {
		return err
	}
	return c.Link(m.Name, md)
}
