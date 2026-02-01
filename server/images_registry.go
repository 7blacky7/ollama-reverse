// Package server - Registry-Operationen (Push, Pull)
//
// Diese Datei enthaelt:
// - PushModel: Model zur Registry hochladen
// - PullModel: Model von Registry herunterladen
// - registryOptions: Optionen fuer Registry-Zugriff
package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

var errInsecureProtocol = errors.New("insecure protocol http")

// registryOptions enthaelt Optionen fuer Registry-Operationen
type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

// PushModel laedt ein Model zur Registry hoch
func PushModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	n := model.ParseName(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Schneller Transfer fuer Models mit Tensor-Layern (viele kleine Blobs)
	if hasTensorLayers(layers) {
		// Rohe Manifest-JSON lesen um Tensor-Metadaten zu erhalten
		manifestPath, err := manifest.PathForName(n)
		if err != nil {
			return err
		}
		manifestJSON, err := os.ReadFile(manifestPath)
		if err != nil {
			return err
		}
		if err := pushWithTransfer(ctx, n, layers, manifestJSON, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	for _, layer := range layers {
		if err := uploadBlob(ctx, n, layer, regOpts, fn); err != nil {
			slog.Info(fmt.Sprintf("error uploading blob: %v", err))
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := n.BaseURL()
	requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	manifestJSON, err := json.Marshal(mf)
	if err != nil {
		return err
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

// PullModel laedt ein Model von der Registry herunter
func PullModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	n := model.ParseName(name)

	// deleteMap fuer Bereinigung ungenutzter Layer aufbauen
	deleteMap := make(map[string]struct{})
	existingMf, err := manifest.ParseNamedManifest(n)
	if errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		slog.Warn("pulling model with bad existing manifest", "name", name, "error", err)
	} else {
		for _, l := range existingMf.Layers {
			deleteMap[l.Digest] = struct{}{}
		}
		if existingMf.Config.Digest != "" {
			deleteMap[existingMf.Config.Digest] = struct{}{}
		}
	}

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	mf, err := pullModelManifest(ctx, n, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Schneller Transfer fuer Models mit Tensor-Layern (viele kleine Blobs)
	if hasTensorLayers(layers) {
		if err := pullWithTransfer(ctx, n, layers, mf, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	skipVerify := make(map[string]bool)
	for _, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			n:       n,
			digest:  layer.Digest,
			regOpts: regOpts,
			fn:      fn,
		})
		if err != nil {
			return err
		}
		skipVerify[layer.Digest] = cacheHit
		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if skipVerify[layer.Digest] {
			continue
		}
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				fp, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					slog.Info(fmt.Sprintf("couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
			}
			return err
		}
	}

	for _, layer := range layers {
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, mf.Config.Digest)

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

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't write to %s", fp))
		return err
	}

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}
