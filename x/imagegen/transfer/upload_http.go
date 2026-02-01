// Package transfer - HTTP-Operationen für Blob-Uploads
//
// Diese Datei enthält die HTTP-spezifischen Operationen für Uploads:
// - Existenzprüfung (HEAD-Requests)
// - Upload-Initialisierung (POST-Requests)
// - Blob-Upload (PUT-Requests)
// - Manifest-Push
//
// Alle Methoden unterstützen automatische Token-Erneuerung bei 401-Fehlern.
package transfer

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
)

// exists prüft ob ein Blob bereits auf dem Server existiert
func (u *uploader) exists(ctx context.Context, blob Blob) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodHead, fmt.Sprintf("%s/v2/%s/blobs/%s", u.baseURL, u.repository, blob.Digest), nil)
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return false, err
	}
	resp.Body.Close()

	// Bei 401 Token erneuern und erneut versuchen
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return false, err
		}
		return u.exists(ctx, blob)
	}

	return resp.StatusCode == http.StatusOK, nil
}

// initUpload initialisiert einen Upload und gibt die Upload-URL zurück
func (u *uploader) initUpload(ctx context.Context, blob Blob) (string, error) {
	endpoint, _ := url.Parse(fmt.Sprintf("%s/v2/%s/blobs/uploads/", u.baseURL, u.repository))
	q := endpoint.Query()
	q.Set("digest", blob.Digest)
	endpoint.RawQuery = q.Encode()

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), nil)
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return "", err
	}
	resp.Body.Close()

	// Bei 401 Token erneuern und erneut versuchen
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return "", err
		}
		return u.initUpload(ctx, blob)
	}

	if resp.StatusCode != http.StatusAccepted {
		return "", fmt.Errorf("init: status %d", resp.StatusCode)
	}

	// Upload-Location aus Header extrahieren
	loc := resp.Header.Get("Docker-Upload-Location")
	if loc == "" {
		loc = resp.Header.Get("Location")
	}
	if loc == "" {
		return "", fmt.Errorf("no upload location")
	}

	// Relative URLs auflösen
	locURL, _ := url.Parse(loc)
	if !locURL.IsAbs() {
		base, _ := url.Parse(u.baseURL)
		locURL = base.ResolveReference(locURL)
	}
	q = locURL.Query()
	q.Set("digest", blob.Digest)
	locURL.RawQuery = q.Encode()

	return locURL.String(), nil
}

// put lädt den Blob-Inhalt hoch
func (u *uploader) put(ctx context.Context, uploadURL string, f *os.File, size int64) (int64, error) {
	pr := &progressReader{reader: f, tracker: u.progress}

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, pr)
	req.ContentLength = size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return pr.n, err
	}
	defer resp.Body.Close()

	// Bei 401 Token erneuern und erneut versuchen
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return pr.n, err
		}
		f.Seek(0, 0)
		u.progress.add(-pr.n)
		return u.put(ctx, uploadURL, f, size)
	}

	// Bei Redirect zu CDN weiterleiten
	if resp.StatusCode == http.StatusTemporaryRedirect {
		return u.putToCDN(ctx, resp, f, size, pr.n)
	}

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return pr.n, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	return pr.n, nil
}

// putToCDN behandelt Redirects zu CDN-Servern
func (u *uploader) putToCDN(ctx context.Context, resp *http.Response, f *os.File, size int64, alreadySent int64) (int64, error) {
	loc, _ := resp.Location()
	f.Seek(0, 0)
	u.progress.add(-alreadySent)
	pr2 := &progressReader{reader: f, tracker: u.progress}

	req2, _ := http.NewRequestWithContext(ctx, http.MethodPut, loc.String(), pr2)
	req2.ContentLength = size
	req2.Header.Set("Content-Type", "application/octet-stream")
	req2.Header.Set("User-Agent", u.userAgent)

	resp2, err := u.client.Do(req2)
	if err != nil {
		return pr2.n, err
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusCreated && resp2.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp2.Body)
		return pr2.n, fmt.Errorf("status %d: %s", resp2.StatusCode, body)
	}
	return pr2.n, nil
}

// pushManifest lädt ein Manifest zur Registry hoch
func (u *uploader) pushManifest(ctx context.Context, repo, ref string, manifest []byte) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", u.baseURL, repo, ref), bytes.NewReader(manifest))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Bei 401 Token erneuern und erneut versuchen
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return err
		}
		return u.pushManifest(ctx, repo, ref, manifest)
	}

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	return nil
}

// progressReader ist ein io.Reader Wrapper der den Fortschritt trackt
type progressReader struct {
	reader  io.Reader
	tracker *progressTracker
	n       int64
}

// Read implementiert io.Reader mit Fortschrittstracking
func (r *progressReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 {
		r.n += int64(n)
		r.tracker.add(int64(n))
	}
	return n, err
}
