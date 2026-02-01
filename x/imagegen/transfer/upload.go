// Package transfer - Upload-Koordination für Blob-Transfers
//
// Diese Datei enthält die Hauptlogik für das Hochladen von Blobs
// zu einer OCI-kompatiblen Registry mit paralleler Verarbeitung.
//
// Hauptfunktionen:
// - upload: Koordiniert den Upload-Prozess mit Existenzprüfung
// - uploader: Hauptstruktur für Upload-Operationen
package transfer

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

// uploader enthält die Konfiguration für Upload-Operationen
type uploader struct {
	client     *http.Client
	baseURL    string
	srcDir     string
	repository string // Repository-Pfad für Blob-URLs (z.B. "library/model")
	token      *string
	getToken   func(context.Context, AuthChallenge) (string, error)
	userAgent  string
	progress   *progressTracker
	logger     *slog.Logger
}

// upload koordiniert das Hochladen mehrerer Blobs mit paralleler Existenzprüfung
func upload(ctx context.Context, opts UploadOptions) error {
	if len(opts.Blobs) == 0 && len(opts.Manifest) == 0 {
		return nil
	}

	token := opts.Token
	u := &uploader{
		client:     cmp.Or(opts.Client, defaultClient),
		baseURL:    opts.BaseURL,
		srcDir:     opts.SrcDir,
		repository: cmp.Or(opts.Repository, "library/_"),
		token:      &token,
		getToken:   opts.GetToken,
		userAgent:  cmp.Or(opts.UserAgent, defaultUserAgent),
		logger:     opts.Logger,
	}

	if len(opts.Blobs) > 0 {
		// Phase 1: Schnelle parallele HEAD-Prüfung welche Blobs hochgeladen werden müssen
		needsUpload, err := u.checkBlobsExist(ctx, opts.Blobs)
		if err != nil {
			return err
		}

		// Filtere auf Blobs die hochgeladen werden müssen
		toUpload, total := filterBlobsForUpload(opts.Blobs, needsUpload)

		if len(toUpload) == 0 {
			if u.logger != nil {
				u.logger.Debug("all blobs exist, nothing to upload")
			}
		} else {
			// Phase 2: Lade Blobs hoch die nicht existieren
			if err := u.uploadBlobs(ctx, toUpload, total, opts.Concurrency, opts.Progress); err != nil {
				return err
			}
		}
	}

	if len(opts.Manifest) > 0 && opts.ManifestRef != "" && opts.Repository != "" {
		return u.pushManifest(ctx, opts.Repository, opts.ManifestRef, opts.Manifest)
	}
	return nil
}

// checkBlobsExist prüft parallel welche Blobs bereits auf dem Server existieren
func (u *uploader) checkBlobsExist(ctx context.Context, blobs []Blob) ([]bool, error) {
	needsUpload := make([]bool, len(blobs))
	sem := semaphore.NewWeighted(128) // Hohe Parallelität für HEAD-Anfragen
	g, gctx := errgroup.WithContext(ctx)

	for i, blob := range blobs {
		g.Go(func() error {
			if err := sem.Acquire(gctx, 1); err != nil {
				return err
			}
			defer sem.Release(1)

			exists, err := u.exists(gctx, blob)
			if err != nil {
				return err
			}
			if !exists {
				needsUpload[i] = true
			} else if u.logger != nil {
				u.logger.Debug("blob exists", "digest", blob.Digest)
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return needsUpload, nil
}

// filterBlobsForUpload filtert Blobs und berechnet die Gesamtgröße
func filterBlobsForUpload(blobs []Blob, needsUpload []bool) ([]Blob, int64) {
	var toUpload []Blob
	var total int64
	for i, blob := range blobs {
		if needsUpload[i] {
			toUpload = append(toUpload, blob)
			total += blob.Size
		}
	}
	return toUpload, total
}

// uploadBlobs lädt die gefilterten Blobs parallel hoch
func (u *uploader) uploadBlobs(ctx context.Context, blobs []Blob, total int64, concurrency int, progressFn func(completed, total int64)) error {
	u.progress = newProgressTracker(total, progressFn)
	concurrency = cmp.Or(concurrency, DefaultUploadConcurrency)
	sem := semaphore.NewWeighted(int64(concurrency))

	g, gctx := errgroup.WithContext(ctx)
	for _, blob := range blobs {
		g.Go(func() error {
			if err := sem.Acquire(gctx, 1); err != nil {
				return err
			}
			defer sem.Release(1)
			return u.upload(gctx, blob)
		})
	}
	return g.Wait()
}

// upload lädt einen einzelnen Blob mit Retry-Logik hoch
func (u *uploader) upload(ctx context.Context, blob Blob) error {
	var lastErr error
	var n int64

	for attempt := range maxRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, time.Second<<uint(attempt-1)); err != nil {
				return err
			}
		}

		var err error
		n, err = u.uploadOnce(ctx, blob)
		if err == nil {
			return nil
		}

		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}

		u.progress.add(-n)
		lastErr = err
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

// uploadOnce führt einen einzelnen Upload-Versuch durch
func (u *uploader) uploadOnce(ctx context.Context, blob Blob) (int64, error) {
	if u.logger != nil {
		u.logger.Debug("uploading blob", "digest", blob.Digest, "size", blob.Size)
	}

	// Upload initialisieren
	uploadURL, err := u.initUpload(ctx, blob)
	if err != nil {
		return 0, err
	}

	// Datei öffnen
	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// PUT Blob
	return u.put(ctx, uploadURL, f, blob.Size)
}
