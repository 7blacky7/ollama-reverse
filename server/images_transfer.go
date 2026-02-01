// Package server - Transfer-Funktionen fuer schnelle Registry-Uebertragung
//
// Diese Datei enthaelt:
// - pullWithTransfer: Schneller Download via x/transfer Package
// - pushWithTransfer: Schneller Upload via x/transfer Package
// - hasTensorLayers: Prueft ob Tensor-Layer vorhanden
// - pullModelManifest: Manifest von Registry laden
package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen/transfer"
)

// hasTensorLayers prueft ob Layer mit Tensor-MediaType vorhanden sind
func hasTensorLayers(layers []manifest.Layer) bool {
	for _, layer := range layers {
		if layer.MediaType == manifest.MediaTypeImageTensor {
			return true
		}
	}
	return false
}

// pullWithTransfer nutzt das x/transfer Package fuer schnelle Downloads
func pullWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, mf *manifest.Manifest, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
		}
	}

	destDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pulling model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	if err := transfer.Download(ctx, transfer.DownloadOptions{
		Blobs:      blobs,
		BaseURL:    baseURL,
		DestDir:    destDir,
		Repository: n.DisplayNamespaceModel(),
		Progress:   progress,
		Token:      regOpts.Token,
		GetToken:   getToken,
		Logger:     slog.Default(),
	}); err != nil {
		return err
	}

	// Manifest schreiben
	fn(api.ProgressResponse{Status: "writing manifest"})
	manifestJSON, err := json.Marshal(mf)
	if err != nil {
		return err
	}

	fp, err := manifest.PathForName(n)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	return os.WriteFile(fp, manifestJSON, 0o644)
}

// pushWithTransfer nutzt das x/transfer Package fuer schnelle Uploads
func pushWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, manifestJSON []byte, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
			From:   layer.From,
		}
	}

	srcDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pushing model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	return transfer.Upload(ctx, transfer.UploadOptions{
		Blobs:       blobs,
		BaseURL:     baseURL,
		SrcDir:      srcDir,
		Progress:    progress,
		Token:       regOpts.Token,
		GetToken:    getToken,
		Logger:      slog.Default(),
		Manifest:    manifestJSON,
		ManifestRef: n.Tag,
		Repository:  n.DisplayNamespaceModel(),
	})
}

// pullModelManifest laedt das Manifest eines Models von der Registry
func pullModelManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, error) {
	requestURL := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m manifest.Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return &m, err
}
