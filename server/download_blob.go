// Download-Blob-Modul: Blob-Download-Logik und Chunk-Verwaltung
// Dieses Modul enthaelt die eigentliche Download-Implementierung.
// Aufgeteilt aus der urspruenglichen download.go (509 LOC)

package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func (b *blobDownload) run(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	defer blobDownloadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	file, err := os.OpenFile(b.Name+"-partial", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	setSparse(file)

	_ = file.Truncate(b.Total)

	directURL, err := func() (*url.URL, error) {
		ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		backoff := newBackoff(10 * time.Second)
		for {
			// shallow clone opts to be used in the closure
			// without affecting the outer opts.
			newOpts := new(registryOptions)
			*newOpts = *opts

			newOpts.CheckRedirect = func(req *http.Request, via []*http.Request) error {
				if len(via) > 10 {
					return errMaxRedirectsExceeded
				}

				// if the hostname is the same, allow the redirect
				if req.URL.Hostname() == requestURL.Hostname() {
					return nil
				}

				// stop at the first redirect that is not
				// the same hostname as the original
				// request.
				return http.ErrUseLastResponse
			}

			resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, nil, nil, newOpts)
			if err != nil {
				slog.Warn("failed to get direct URL; backing off and retrying", "err", err)
				if err := backoff(ctx); err != nil {
					return nil, err
				}
				continue
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusTemporaryRedirect && resp.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
			}
			return resp.Location()
		}
	}()
	if err != nil {
		return err
	}

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numDownloadParts)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed.Load() == part.Size {
			continue
		}

		g.Go(func() error {
			var err error
			for try := 0; try < maxRetries; try++ {
				w := io.NewOffsetWriter(file, part.StartsAt())
				err = b.downloadChunk(inner, directURL, w, part)
				switch {
				case errors.Is(err, context.Canceled), errors.Is(err, syscall.ENOSPC):
					// return immediately if the context is canceled or the device is out of space
					return err
				case errors.Is(err, errPartStalled):
					try--
					continue
				case err != nil:
					sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
					slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
					time.Sleep(sleep)
					continue
				default:
					return nil
				}
			}

			return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// explicitly close the file so we can rename it
	if err := file.Close(); err != nil {
		return err
	}

	for i := range b.Parts {
		if err := os.Remove(file.Name() + "-" + strconv.Itoa(i)); err != nil {
			return err
		}
	}

	if err := os.Rename(file.Name(), b.Name); err != nil {
		return err
	}

	return nil
}

func (b *blobDownload) downloadChunk(ctx context.Context, requestURL *url.URL, w io.Writer, part *blobDownloadPart) error {
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", part.StartsAt(), part.StopsAt()-1))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		n, err := io.CopyN(w, io.TeeReader(resp.Body, part), part.Size-part.Completed.Load())
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, io.ErrUnexpectedEOF) {
			// rollback progress
			b.Completed.Add(-n)
			return err
		}

		part.Completed.Add(n)
		if err := b.writePart(part.Name(), part); err != nil {
			return err
		}

		// return nil or context.Canceled or UnexpectedEOF (resumable)
		return err
	})

	g.Go(func() error {
		ticker := time.NewTicker(time.Second)
		for {
			select {
			case <-ticker.C:
				if part.Completed.Load() >= part.Size {
					return nil
				}

				part.lastUpdatedMu.Lock()
				lastUpdated := part.lastUpdated
				part.lastUpdatedMu.Unlock()

				if !lastUpdated.IsZero() && time.Since(lastUpdated) > 30*time.Second {
					const msg = "%s part %d stalled; retrying. If this persists, press ctrl-c to exit, then 'ollama pull' to find a faster connection."
					slog.Info(fmt.Sprintf(msg, b.Digest[7:19], part.N))
					// reset last updated
					part.lastUpdatedMu.Lock()
					part.lastUpdated = time.Time{}
					part.lastUpdatedMu.Unlock()
					return errPartStalled
				}
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})

	return g.Wait()
}

type downloadOpts struct {
	n       model.Name
	digest  string
	regOpts *registryOptions
	fn      func(api.ProgressResponse)
}

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, opts downloadOpts) (cacheHit bool, _ error) {
	if opts.digest == "" {
		return false, fmt.Errorf(("%s: %s"), opts.n.DisplayNamespaceModel(), "digest is empty")
	}

	fp, err := manifest.BlobsPath(opts.digest)
	if err != nil {
		return false, err
	}

	fi, err := os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return false, err
	default:
		opts.fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pulling %s", opts.digest[7:19]),
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return true, nil
	}

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.n.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.n.DisplayNamespaceModel(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return false, err
		}

		//nolint:contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	return false, download.Wait(ctx, opts.fn)
}
