// Package server - Blob Upload Vorbereitung
// Beinhaltet: Prepare Methode
// Initialisiert Upload-Parts und prueft Mount-Moeglichkeit
package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// Prepare bereitet den Blob-Upload vor
// Prueft ob Blob gemountet werden kann und partitioniert bei Bedarf
func (b *blobUpload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	p, err := manifest.BlobsPath(b.Digest)
	if err != nil {
		return err
	}

	if b.From != "" {
		values := requestURL.Query()
		values.Add("mount", b.Digest)
		values.Add("from", model.ParseName(b.From).DisplayNamespaceModel())
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, http.MethodPost, requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	fi, err := os.Stat(p)
	if err != nil {
		return err
	}

	b.Total = fi.Size()

	// http.StatusCreated indicates a blob has been mounted
	// ref: https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount
	if resp.StatusCode == http.StatusCreated {
		b.Completed.Store(b.Total)
		b.done = true
		return nil
	}

	size := b.Total / numUploadParts
	switch {
	case size < minUploadPartSize:
		size = minUploadPartSize
	case size > maxUploadPartSize:
		size = maxUploadPartSize
	}

	var offset int64
	for offset < fi.Size() {
		if offset+size > fi.Size() {
			size = fi.Size() - offset
		}

		// set part.N to the current number of parts
		b.Parts = append(b.Parts, blobUploadPart{N: len(b.Parts), Offset: offset, Size: size})
		offset += size
	}

	if len(b.Parts) > 0 {
		slog.Info(fmt.Sprintf("uploading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size)))
	}

	requestURL, err = url.Parse(location)
	if err != nil {
		return err
	}

	b.nextURL = make(chan *url.URL, 1)
	b.nextURL <- requestURL
	return nil
}
