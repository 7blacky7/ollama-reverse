// Package transfer - I/O-Operationen für Blob-Downloads
//
// Diese Datei enthält die HTTP- und Datei-Operationen für Downloads:
// - downloadOnce: Einzelner Download-Versuch
// - save: Speichert Blob auf Festplatte mit Hash-Verifikation
// - copy: Kopiert Daten mit Stall/Slow-Erkennung
// - resolve: URL-Auflösung mit Redirect-Handling
//
// Unterstützt automatische Token-Erneuerung und CDN-Redirects.
package transfer

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// downloadOnce führt einen einzelnen Download-Versuch durch
func (d *downloader) downloadOnce(ctx context.Context, blob Blob) (int64, error) {
	if d.logger != nil {
		d.logger.Debug("downloading blob", "digest", blob.Digest, "size", blob.Size)
	}

	baseURL, _ := url.Parse(d.baseURL)
	u, err := d.resolve(ctx, fmt.Sprintf("%s/v2/%s/blobs/%s", d.baseURL, d.repository, blob.Digest))
	if err != nil {
		return 0, err
	}

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	req.Header.Set("User-Agent", d.userAgent)
	// Auth nur für gleichen Host hinzufügen (nicht für CDN)
	if u.Host == baseURL.Host && *d.token != "" {
		req.Header.Set("Authorization", "Bearer "+*d.token)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("status %d", resp.StatusCode)
	}

	return d.save(ctx, blob, resp.Body)
}

// save speichert den Blob auf der Festplatte mit Hash-Verifikation
func (d *downloader) save(ctx context.Context, blob Blob, r io.Reader) (int64, error) {
	dest := filepath.Join(d.destDir, digestToPath(blob.Digest))
	tmp := dest + ".tmp"
	os.MkdirAll(filepath.Dir(dest), 0o755)

	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	setSparse(f)

	h := sha256.New()
	n, err := d.copy(ctx, f, r, h)
	if err != nil {
		os.Remove(tmp)
		return n, err
	}
	f.Close()

	// Hash verifizieren
	if got := fmt.Sprintf("sha256:%x", h.Sum(nil)); got != blob.Digest {
		os.Remove(tmp)
		return n, fmt.Errorf("digest mismatch")
	}
	// Größe verifizieren
	if n != blob.Size {
		os.Remove(tmp)
		return n, fmt.Errorf("size mismatch")
	}
	return n, os.Rename(tmp, dest)
}

// copy kopiert Daten mit Stall- und Slow-Erkennung
func (d *downloader) copy(ctx context.Context, dst io.Writer, src io.Reader, h io.Writer) (int64, error) {
	var n int64
	var lastRead atomic.Int64
	lastRead.Store(time.Now().UnixNano())
	start := time.Now()

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	// Hintergrund-Goroutine für Stall/Slow-Erkennung
	go d.monitorCopyProgress(ctx, cancel, &lastRead, &n, start)

	buf := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			if c := context.Cause(ctx); c != nil {
				return n, c
			}
			return n, err
		}

		nr, err := src.Read(buf)
		if nr > 0 {
			lastRead.Store(time.Now().UnixNano())
			dst.Write(buf[:nr])
			h.Write(buf[:nr])
			d.progress.add(int64(nr))
			n += int64(nr)
		}
		if err == io.EOF {
			return n, nil
		}
		if err != nil {
			return n, err
		}
	}
}

// monitorCopyProgress überwacht den Kopierfortschritt auf Stall/Slow
func (d *downloader) monitorCopyProgress(ctx context.Context, cancel context.CancelCauseFunc, lastRead *atomic.Int64, n *int64, start time.Time) {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			// Stall-Erkennung
			if time.Since(time.Unix(0, lastRead.Load())) > d.stallTimeout {
				cancel(errStalled)
				return
			}
			// Slow-Erkennung (nach 5 Sekunden)
			if e := time.Since(start); e > 5*time.Second {
				if m := d.speeds.median(); m > 0 && float64(*n)/e.Seconds() < m*0.1 {
					cancel(errSlow)
					return
				}
			}
		}
	}
}

// resolve löst eine URL auf und folgt Redirects bis zum finalen Ziel
func (d *downloader) resolve(ctx context.Context, rawURL string) (*url.URL, error) {
	u, _ := url.Parse(rawURL)
	const maxRedirects = 10

	for range maxRedirects {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		req.Header.Set("User-Agent", d.userAgent)
		if *d.token != "" {
			req.Header.Set("Authorization", "Bearer "+*d.token)
		}

		resp, err := d.client.Do(req)
		if err != nil {
			return nil, err
		}
		resp.Body.Close()

		switch resp.StatusCode {
		case http.StatusOK:
			return u, nil
		case http.StatusUnauthorized:
			if d.getToken == nil {
				return nil, fmt.Errorf("unauthorized")
			}
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if *d.token, err = d.getToken(ctx, ch); err != nil {
				return nil, err
			}
		case http.StatusTemporaryRedirect, http.StatusFound, http.StatusMovedPermanently:
			loc, _ := resp.Location()
			// Bei anderem Host (CDN) sofort zurückgeben
			if loc.Host != u.Host {
				return loc, nil
			}
			u = loc
		default:
			return nil, fmt.Errorf("status %d", resp.StatusCode)
		}
	}
	return nil, fmt.Errorf("too many redirects")
}
