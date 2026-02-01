// Package blob implements a content-addressable disk cache for blobs and
// manifests.
//
// Modul: cache.go - DiskCache Kernfunktionen
// Enthaelt: DiskCache Struktur, Open, Put, Get, Import und Hilfsfunktionen
package blob

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Entry contains metadata about a blob in the cache.
type Entry struct {
	Digest Digest
	Size   int64
	Time   time.Time // when added to the cache
}

// DiskCache caches blobs and manifests on disk.
//
// The cache is rooted at a directory, which is created if it does not exist.
//
// Blobs are stored in the "blobs" subdirectory, and manifests are stored in the
// "manifests" subdirectory. A example directory structure might look like:
//
//	<dir>/
//	  blobs/
//	    sha256-<digest> - <blob data>
//	  manifests/
//	    <host>/
//	      <namespace>/
//	        <name>/
//	          <tag> - <manifest data>
//
// The cache is safe for concurrent use.
//
// Name casing is preserved in the cache, but is not significant when resolving
// names. For example, "Foo" and "foo" are considered the same name.
//
// The cache is not safe for concurrent use. It guards concurrent writes, but
// does not prevent duplicated effort. Because blobs are immutable, duplicate
// writes should result in the same file being written to disk.
type DiskCache struct {
	// Dir specifies the top-level directory where blobs and manifest
	// pointers are stored.
	dir string
	now func() time.Time

	testHookBeforeFinalWrite func(f *os.File)
}

// PutBytes is a convenience function for c.Put(d, strings.NewReader(s), int64(len(s))).
func PutBytes[S string | []byte](c *DiskCache, d Digest, data S) error {
	return c.Put(d, bytes.NewReader([]byte(data)), int64(len(data)))
}

// Open opens a cache rooted at the given directory. If the directory does not
// exist, it is created. If the directory is not a directory, an error is
// returned.
func Open(dir string) (*DiskCache, error) {
	if dir == "" {
		return nil, errors.New("blob: empty directory name")
	}

	info, err := os.Stat(dir)
	if err == nil && !info.IsDir() {
		return nil, fmt.Errorf("%q is not a directory", dir)
	}
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return nil, err
	}

	subdirs := []string{"blobs", "manifests"}
	for _, subdir := range subdirs {
		if err := os.MkdirAll(filepath.Join(dir, subdir), 0o777); err != nil {
			return nil, err
		}
	}

	// TODO(bmizerany): support shards
	c := &DiskCache{
		dir: dir,
		now: time.Now,
	}
	return c, nil
}

// Put writes a new blob to the cache, identified by its digest. The operation
// reads content from r, which must precisely match both the specified size and
// digest.
//
// Concurrent write safety is achieved through file locking. The implementation
// guarantees write integrity by enforcing size limits and content validation
// before allowing the file to reach its final state.
func (c *DiskCache) Put(d Digest, r io.Reader, size int64) error {
	return c.copyNamedFile(c.GetFile(d), r, d, size)
}

// Import imports a blob from the provided reader into the cache. It reads the
// entire content of the reader, calculates its digest, and stores it in the
// cache.
//
// Import should be considered unsafe for use with untrusted data, such as data
// read from a network. The caller is responsible for ensuring the integrity of
// the data being imported.
func (c *DiskCache) Import(r io.Reader, size int64) (Digest, error) {
	// users that want to change the temp dir can set TEMPDIR.
	f, err := os.CreateTemp("", "blob-")
	if err != nil {
		return Digest{}, err
	}
	defer os.Remove(f.Name())

	// Copy the blob to a temporary file.
	h := sha256.New()
	r = io.TeeReader(r, h)
	n, err := io.Copy(f, r)
	if err != nil {
		return Digest{}, err
	}
	if n != size {
		return Digest{}, fmt.Errorf("blob: expected %d bytes, got %d", size, n)
	}

	// Check the digest.
	var d Digest
	h.Sum(d.sum[:0])
	if err := f.Close(); err != nil {
		return Digest{}, err
	}
	name := c.GetFile(d)
	// Rename the temporary file to the final file.
	if err := os.Rename(f.Name(), name); err != nil {
		return Digest{}, err
	}
	os.Chtimes(name, c.now(), c.now()) // mainly for tests
	return d, nil
}

// Get retrieves a blob from the cache using the provided digest. The operation
// fails if the digest is malformed or if any errors occur during blob
// retrieval.
func (c *DiskCache) Get(d Digest) (Entry, error) {
	name := c.GetFile(d)
	info, err := os.Stat(name)
	if err != nil {
		return Entry{}, err
	}
	if info.Size() == 0 {
		return Entry{}, fs.ErrNotExist
	}
	return Entry{
		Digest: d,
		Size:   info.Size(),
		Time:   info.ModTime(),
	}, nil
}

// GetFile returns the absolute path to the file, in the cache, for the given
// digest. It does not check if the file exists.
//
// The returned path should not be stored, used outside the lifetime of the
// cache, or interpreted in any way.
func (c *DiskCache) GetFile(d Digest) string {
	filename := fmt.Sprintf("sha256-%x", d.sum)
	return absJoin(c.dir, "blobs", filename)
}

func splitNameDigest(s string) (name, digest string) {
	i := strings.LastIndexByte(s, '@')
	if i < 0 {
		return s, ""
	}
	return s[:i], s[i+1:]
}

func absJoin(pp ...string) string {
	abs, err := filepath.Abs(filepath.Join(pp...))
	if err != nil {
		panic(err) // this should never happen
	}
	return abs
}
