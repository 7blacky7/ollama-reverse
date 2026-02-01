// Package server - Upload Typen und Konstanten
// Beinhaltet: blobUpload struct, blobUploadPart, progressWriter, Konstanten
// Abgetrennt aus upload.go fuer bessere Wartbarkeit
package server

import (
	"context"
	"hash"
	"os"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
)

// blobUploadManager verwaltet aktive Blob-Uploads global
var blobUploadManager sync.Map

// Upload-Konstanten fuer Partitionierung
const (
	numUploadParts          = 16
	minUploadPartSize int64 = 100 * format.MegaByte
	maxUploadPartSize int64 = 1000 * format.MegaByte
)

// blobUpload repraesentiert einen laufenden Blob-Upload
type blobUpload struct {
	manifest.Layer

	Total     int64
	Completed atomic.Int64

	Parts []blobUploadPart

	nextURL chan *url.URL

	context.CancelFunc

	file *os.File

	done       bool
	err        error
	references atomic.Int32
}

// blobUploadPart repraesentiert einen Teil eines Blob-Uploads
type blobUploadPart struct {
	// N is the part number
	N      int
	Offset int64
	Size   int64
	hash.Hash
}

// progressWriter verfolgt den Upload-Fortschritt
type progressWriter struct {
	written int64
	*blobUpload
}

// Write implementiert io.Writer fuer Fortschrittsverfolgung
func (p *progressWriter) Write(b []byte) (n int, err error) {
	n = len(b)
	p.written += int64(n)
	p.Completed.Add(int64(n))
	return n, nil
}

// Rollback setzt den Fortschritt zurueck bei Fehlern
func (p *progressWriter) Rollback() {
	p.Completed.Add(-p.written)
	p.written = 0
}
