// Package ollama - Chunksum-Handling für große Layer
//
// Diese Datei enthält:
// - chunksum-Struct für Chunk-Metadaten
// - chunksums() für iterative Chunk-Auflösung
// - Unterstützung für große Layer-Downloads
package ollama

import (
	"bufio"
	"context"
	"fmt"
	"iter"

	"github.com/ollama/ollama/server/internal/cache/blob"
)

// chunksum enthält URL, Chunk-Range und Digest für einen Download-Chunk
type chunksum struct {
	URL    string
	Chunk  blob.Chunk
	Digest blob.Digest
}

// chunksums liefert Chunksums für einen Layer
// Kleine Layer bekommen einen einzelnen Chunk, große werden aufgeteilt
func (r *Registry) chunksums(ctx context.Context, name string, l *Layer) iter.Seq2[chunksum, error] {
	return func(yield func(chunksum, error) bool) {
		scheme, n, _, err := r.parseNameExtended(name)
		if err != nil {
			yield(chunksum{}, err)
			return
		}

		if l.Size < r.maxChunkingThreshold() {
			// Kleine Layer in einem Request herunterladen
			cs := chunksum{
				URL: fmt.Sprintf("%s://%s/v2/%s/%s/blobs/%s",
					scheme,
					n.Host(),
					n.Namespace(),
					n.Model(),
					l.Digest,
				),
				Chunk:  blob.Chunk{Start: 0, End: l.Size - 1},
				Digest: l.Digest,
			}
			yield(cs, nil)
			return
		}

		// Chunksums vom Server abrufen
		// Format:
		//   GET /v2/<namespace>/<model>/chunksums/<digest>
		//   Response:
		//     Content-Location: <blobURL>
		//     <digest> <start>-<end>
		//     ...
		chunksumsURL := fmt.Sprintf("%s://%s/v2/%s/%s/chunksums/%s",
			scheme,
			n.Host(),
			n.Namespace(),
			n.Model(),
			l.Digest,
		)

		req, err := r.newRequest(ctx, "GET", chunksumsURL, nil)
		if err != nil {
			yield(chunksum{}, err)
			return
		}
		res, err := sendRequest(r.client(), req)
		if err != nil {
			yield(chunksum{}, err)
			return
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			err := fmt.Errorf("chunksums: unexpected status code %d", res.StatusCode)
			yield(chunksum{}, err)
			return
		}
		blobURL := res.Header.Get("Content-Location")

		s := bufio.NewScanner(res.Body)
		s.Split(bufio.ScanWords)
		for {
			if !s.Scan() {
				if s.Err() != nil {
					yield(chunksum{}, s.Err())
				}
				return
			}
			d, err := blob.ParseDigest(s.Bytes())
			if err != nil {
				yield(chunksum{}, fmt.Errorf("invalid digest: %q", s.Bytes()))
				return
			}

			if !s.Scan() {
				err := s.Err()
				if err == nil {
					err = fmt.Errorf("missing chunk range for digest %s", d)
				}
				yield(chunksum{}, err)
				return
			}
			chunk, err := parseChunk(s.Bytes())
			if err != nil {
				yield(chunksum{}, fmt.Errorf("invalid chunk range for digest %s: %q", d, s.Bytes()))
				return
			}

			cs := chunksum{
				URL:    blobURL,
				Chunk:  chunk,
				Digest: d,
			}
			if !yield(cs, nil) {
				return
			}
		}
	}
}
