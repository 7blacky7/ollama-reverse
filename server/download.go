// Download-Modul: Hauptstruktur und Typen fuer Blob-Downloads
// Dieses Modul verwaltet die Download-Strukturen und Part-Verwaltung.
// Aufgeteilt aus der urspruenglichen download.go (509 LOC)
// Siehe auch: download_blob.go

package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

const maxRetries = 6

var (
	errMaxRetriesExceeded   = errors.New("max retries exceeded")
	errPartStalled          = errors.New("part stalled")
	errMaxRedirectsExceeded = errors.New("maximum redirects exceeded (10) for directURL")
)

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	Total     int64
	Completed atomic.Int64

	Parts []*blobDownloadPart

	context.CancelFunc

	done       chan struct{}
	err        error
	references atomic.Int32
}

type blobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed atomic.Int64

	lastUpdatedMu sync.Mutex
	lastUpdated   time.Time

	*blobDownload `json:"-"`
}

type jsonBlobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed int64
}

func (p *blobDownloadPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(jsonBlobDownloadPart{
		N:         p.N,
		Offset:    p.Offset,
		Size:      p.Size,
		Completed: p.Completed.Load(),
	})
}

func (p *blobDownloadPart) UnmarshalJSON(b []byte) error {
	var j jsonBlobDownloadPart
	if err := json.Unmarshal(b, &j); err != nil {
		return err
	}
	*p = blobDownloadPart{
		N:      j.N,
		Offset: j.Offset,
		Size:   j.Size,
	}
	p.Completed.Store(j.Completed)
	return nil
}

const (
	numDownloadParts          = 16
	minDownloadPartSize int64 = 100 * format.MegaByte
	maxDownloadPartSize int64 = 1000 * format.MegaByte
)

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
}

func (p *blobDownloadPart) StartsAt() int64 {
	return p.Offset + p.Completed.Load()
}

func (p *blobDownloadPart) StopsAt() int64 {
	return p.Offset + p.Size
}

func (p *blobDownloadPart) Write(b []byte) (n int, err error) {
	n = len(b)
	p.blobDownload.Completed.Add(int64(n))
	p.lastUpdatedMu.Lock()
	p.lastUpdated = time.Now()
	p.lastUpdatedMu.Unlock()
	return n, nil
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	partFilePaths, err := filepath.Glob(b.Name + "-partial-*")
	if err != nil {
		return err
	}

	b.done = make(chan struct{})

	for _, partFilePath := range partFilePaths {
		part, err := b.readPart(partFilePath)
		if err != nil {
			return err
		}

		b.Total += part.Size
		b.Completed.Add(part.Completed.Load())
		b.Parts = append(b.Parts, part)
	}

	if len(b.Parts) == 0 {
		resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		b.Total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

		size := b.Total / numDownloadParts
		switch {
		case size < minDownloadPartSize:
			size = minDownloadPartSize
		case size > maxDownloadPartSize:
			size = maxDownloadPartSize
		}

		var offset int64
		for offset < b.Total {
			if offset+size > b.Total {
				size = b.Total - offset
			}

			if err := b.newPart(offset, size); err != nil {
				return err
			}

			offset += size
		}
	}

	if len(b.Parts) > 0 {
		slog.Info(fmt.Sprintf("downloading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size)))
	}

	return nil
}

func (b *blobDownload) Run(ctx context.Context, requestURL *url.URL, opts *registryOptions) {
	defer close(b.done)
	b.err = b.run(ctx, requestURL, opts)
}

func newBackoff(maxBackoff time.Duration) func(ctx context.Context) error {
	var n int
	return func(ctx context.Context) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		n++

		// n^2 backoff timer is a little smoother than the
		// common choice of 2^n.
		d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
		// Randomize the delay between 0.5-1.5 x msec, in order
		// to prevent accidental "thundering herd" problems.
		d = time.Duration(float64(d) * (rand.Float64() + 0.5))
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			return nil
		}
	}
}

func (b *blobDownload) newPart(offset, size int64) error {
	part := blobDownloadPart{blobDownload: b, Offset: offset, Size: size, N: len(b.Parts)}
	if err := b.writePart(part.Name(), &part); err != nil {
		return err
	}

	b.Parts = append(b.Parts, &part)
	return nil
}

func (b *blobDownload) readPart(partName string) (*blobDownloadPart, error) {
	var part blobDownloadPart
	partFile, err := os.Open(partName)
	if err != nil {
		return nil, err
	}
	defer partFile.Close()

	if err := json.NewDecoder(partFile).Decode(&part); err != nil {
		return nil, err
	}

	part.blobDownload = b
	return &part, nil
}

func (b *blobDownload) writePart(partName string, part *blobDownloadPart) error {
	partFile, err := os.OpenFile(partName, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer partFile.Close()

	return json.NewEncoder(partFile).Encode(part)
}

func (b *blobDownload) acquire() {
	b.references.Add(1)
}

func (b *blobDownload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobDownload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	b.acquire()
	defer b.release()

	ticker := time.NewTicker(60 * time.Millisecond)
	for {
		select {
		case <-b.done:
			return b.err
		case <-ticker.C:
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("pulling %s", b.Digest[7:19]),
				Digest:    b.Digest,
				Total:     b.Total,
				Completed: b.Completed.Load(),
			})
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
