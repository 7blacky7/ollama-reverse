// Package server - Blob Upload Ausfuehrung
// Beinhaltet: Run, acquire, release, Wait Methoden
// Verwaltet parallele Part-Uploads und Fortschrittsverfolgung
package server

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
)

// Run fuehrt den Blob-Upload durch
// Laedt Parts parallel hoch und finalisiert mit Checksum
func (b *blobUpload) Run(ctx context.Context, opts *registryOptions) {
	defer blobUploadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	p, err := manifest.BlobsPath(b.Digest)
	if err != nil {
		b.err = err
		return
	}

	b.file, err = os.Open(p)
	if err != nil {
		b.err = err
		return
	}
	defer b.file.Close()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range b.Parts {
		part := &b.Parts[i]
		select {
		case <-inner.Done():
		case requestURL := <-b.nextURL:
			g.Go(func() error {
				var err error
				for try := range maxRetries {
					err = b.uploadPart(inner, http.MethodPatch, requestURL, part, opts)
					switch {
					case errors.Is(err, context.Canceled):
						return err
					case errors.Is(err, errMaxRetriesExceeded):
						return err
					case err != nil:
						sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
						slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
						time.Sleep(sleep)
						continue
					}

					return nil
				}

				return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
			})
		}
	}

	if err := g.Wait(); err != nil {
		b.err = err
		return
	}

	requestURL := <-b.nextURL

	// calculate md5 checksum and add it to the commit request
	md5sum := md5.New()
	for _, part := range b.Parts {
		md5sum.Write(part.Sum(nil))
	}

	values := requestURL.Query()
	values.Add("digest", b.Digest)
	values.Add("etag", fmt.Sprintf("%x-%d", md5sum.Sum(nil), len(b.Parts)))
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	for try := range maxRetries {
		var resp *http.Response
		resp, err = makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, nil, opts)
		if errors.Is(err, context.Canceled) {
			break
		} else if err != nil {
			sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
			slog.Info(fmt.Sprintf("%s complete upload attempt %d failed: %v, retrying in %s", b.Digest[7:19], try, err, sleep))
			time.Sleep(sleep)
			continue
		}
		defer resp.Body.Close()
		break
	}

	b.err = err
	b.done = true
}

// acquire erhoeht die Referenz-Zaehlung
func (b *blobUpload) acquire() {
	b.references.Add(1)
}

// release verringert die Referenz-Zaehlung
// Bricht Upload ab wenn keine Referenzen mehr existieren
func (b *blobUpload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

// Wait wartet auf Upload-Abschluss und meldet Fortschritt
func (b *blobUpload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	b.acquire()
	defer b.release()

	ticker := time.NewTicker(60 * time.Millisecond)
	for {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}

		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", b.Digest[7:19]),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

		if b.done || b.err != nil {
			return b.err
		}
	}
}
