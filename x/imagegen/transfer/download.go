// Package transfer - Download-Koordination für Blob-Transfers
//
// Diese Datei enthält die Hauptlogik für das Herunterladen von Blobs
// von einer OCI-kompatiblen Registry mit paralleler Verarbeitung.
//
// Hauptfunktionen:
// - download: Koordiniert den Download-Prozess mit Resume-Unterstützung
// - downloader: Hauptstruktur für Download-Operationen
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

var (
	// errStalled wird ausgelöst wenn der Download ins Stocken gerät
	errStalled = errors.New("download stalled")
	// errSlow wird ausgelöst wenn der Download zu langsam ist
	errSlow = errors.New("download too slow")
)

// downloader enthält die Konfiguration für Download-Operationen
type downloader struct {
	client       *http.Client
	baseURL      string
	destDir      string
	repository   string // Repository-Pfad für Blob-URLs (z.B. "library/model")
	token        *string
	getToken     func(context.Context, AuthChallenge) (string, error)
	userAgent    string
	stallTimeout time.Duration
	progress     *progressTracker
	speeds       *speedTracker
	logger       *slog.Logger
}

// download koordiniert das Herunterladen mehrerer Blobs mit Resume-Unterstützung
func download(ctx context.Context, opts DownloadOptions) error {
	if len(opts.Blobs) == 0 {
		return nil
	}

	// Gesamtgröße berechnen (für genaue Fortschrittsanzeige bei Resume)
	var total int64
	for _, b := range opts.Blobs {
		total += b.Size
	}

	// Bereits heruntergeladene Blobs filtern und abgeschlossene Bytes tracken
	blobs, alreadyCompleted := filterDownloadedBlobs(opts.Blobs, opts.DestDir, opts.Logger)
	if len(blobs) == 0 {
		return nil
	}

	token := opts.Token
	progress := newProgressTracker(total, opts.Progress)
	progress.add(alreadyCompleted) // Bereits heruntergeladene Bytes sofort melden

	d := &downloader{
		client:       cmp.Or(opts.Client, defaultClient),
		baseURL:      opts.BaseURL,
		destDir:      opts.DestDir,
		repository:   cmp.Or(opts.Repository, "library/_"),
		token:        &token,
		getToken:     opts.GetToken,
		userAgent:    cmp.Or(opts.UserAgent, defaultUserAgent),
		stallTimeout: cmp.Or(opts.StallTimeout, defaultStallTimeout),
		progress:     progress,
		speeds:       &speedTracker{},
		logger:       opts.Logger,
	}

	return d.downloadAll(ctx, blobs, opts.Concurrency)
}

// filterDownloadedBlobs filtert bereits vollständig heruntergeladene Blobs
func filterDownloadedBlobs(blobs []Blob, destDir string, logger *slog.Logger) ([]Blob, int64) {
	var remaining []Blob
	var alreadyCompleted int64

	for _, b := range blobs {
		fi, _ := os.Stat(filepath.Join(destDir, digestToPath(b.Digest)))
		if fi != nil && fi.Size() == b.Size {
			if logger != nil {
				logger.Debug("blob already exists", "digest", b.Digest, "size", b.Size)
			}
			alreadyCompleted += b.Size
			continue
		}
		remaining = append(remaining, b)
	}
	return remaining, alreadyCompleted
}

// downloadAll lädt alle Blobs parallel herunter
func (d *downloader) downloadAll(ctx context.Context, blobs []Blob, concurrency int) error {
	concurrency = cmp.Or(concurrency, DefaultDownloadConcurrency)
	sem := semaphore.NewWeighted(int64(concurrency))

	g, ctx := errgroup.WithContext(ctx)
	for _, blob := range blobs {
		g.Go(func() error {
			if err := sem.Acquire(ctx, 1); err != nil {
				return err
			}
			defer sem.Release(1)
			return d.download(ctx, blob)
		})
	}
	return g.Wait()
}

// download lädt einen einzelnen Blob mit Retry-Logik herunter
func (d *downloader) download(ctx context.Context, blob Blob) error {
	var lastErr error
	var slowRetries int
	attempt := 0

	for attempt < maxRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, time.Second<<uint(attempt-1)); err != nil {
				return err
			}
		}

		start := time.Now()
		n, err := d.downloadOnce(ctx, blob)
		if err == nil {
			// Geschwindigkeit aufzeichnen für adaptive Erkennung
			if s := time.Since(start).Seconds(); s > 0 {
				d.speeds.record(float64(blob.Size) / s)
			}
			return nil
		}

		d.progress.add(-n) // Rollback

		switch {
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			return err
		case errors.Is(err, errStalled):
			// Stall-Retries nicht gegen das Limit zählen
		case errors.Is(err, errSlow):
			if slowRetries++; slowRetries >= 3 {
				attempt++ // Nur nach 3 langsamen Retries zählen
			}
		default:
			attempt++
		}
		lastErr = err
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

// Konstante für Stall-Timeout
const defaultStallTimeout = 10 * time.Second
