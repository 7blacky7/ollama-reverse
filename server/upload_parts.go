// Package server - Upload Part Handler
// Beinhaltet: uploadPart, uploadBlob Funktionen
// Verarbeitet einzelne Parts mit Redirect und Auth-Handling
package server

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// uploadPart laedt einen einzelnen Part hoch
// Behandelt Redirects und Authentifizierung automatisch
func (b *blobUpload) uploadPart(ctx context.Context, method string, requestURL *url.URL, part *blobUploadPart, opts *registryOptions) error {
	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", strconv.FormatInt(part.Size, 10))

	if method == http.MethodPatch {
		headers.Set("X-Redirect-Uploads", "1")
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", part.Offset, part.Offset+part.Size-1))
	}

	sr := io.NewSectionReader(b.file, part.Offset, part.Size)

	md5sum := md5.New()
	w := &progressWriter{blobUpload: b}

	resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(sr, io.MultiWriter(w, md5sum)), opts)
	if err != nil {
		w.Rollback()
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	nextURL, err := url.Parse(location)
	if err != nil {
		w.Rollback()
		return err
	}

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		w.Rollback()
		b.nextURL <- nextURL

		redirectURL, err := resp.Location()
		if err != nil {
			return err
		}

		// retry uploading to the redirect URL
		for try := range maxRetries {
			err = b.uploadPart(ctx, http.MethodPut, redirectURL, part, &registryOptions{})
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

	case resp.StatusCode == http.StatusUnauthorized:
		w.Rollback()
		challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
		token, err := getAuthorizationToken(ctx, challenge, requestURL.Host)
		if err != nil {
			return err
		}

		opts.Token = token
		fallthrough
	case resp.StatusCode >= http.StatusBadRequest:
		w.Rollback()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		return fmt.Errorf("http status %s: %s", resp.Status, body)
	}

	if method == http.MethodPatch {
		b.nextURL <- nextURL
	}

	part.Hash = md5sum
	return nil
}

// uploadBlob laedt einen Blob zur Registry hoch
// Prueft ob Blob bereits existiert und startet Upload wenn noetig
func uploadBlob(ctx context.Context, n model.Name, layer manifest.Layer, opts *registryOptions, fn func(api.ProgressResponse)) error {
	requestURL := n.BaseURL()
	requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "blobs", layer.Digest)

	resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		defer resp.Body.Close()
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", layer.Digest[7:19]),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})

		return nil
	}

	data, ok := blobUploadManager.LoadOrStore(layer.Digest, &blobUpload{Layer: layer})
	upload := data.(*blobUpload)
	if !ok {
		requestURL := n.BaseURL()
		requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "blobs/uploads/")
		if err := upload.Prepare(ctx, requestURL, opts); err != nil {
			blobUploadManager.Delete(layer.Digest)
			return err
		}

		//nolint:contextcheck
		go upload.Run(context.Background(), opts)
	}

	return upload.Wait(ctx, fn)
}
