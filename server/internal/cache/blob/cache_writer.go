// Package blob implements a content-addressable disk cache for blobs and
// manifests.
//
// Modul: cache_writer.go - checkWriter und copyNamedFile
// Enthaelt: checkWriter Struktur und Methoden, copyNamedFile Funktion
package blob

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"hash"
	"io"
	"os"
)

type checkWriter struct {
	size int64
	d    Digest
	f    *os.File
	h    hash.Hash

	w   io.Writer // underlying writer; set by creator
	n   int64
	err error

	testHookBeforeFinalWrite func(*os.File)
}

func (w *checkWriter) seterr(err error) error {
	if w.err == nil {
		w.err = err
	}
	return err
}

// Write writes p to the underlying hash and writer. The last write to the
// underlying writer is guaranteed to be the last byte of p as verified by the
// hash.
func (w *checkWriter) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}

	_, err := w.h.Write(p)
	if err != nil {
		return 0, w.seterr(err)
	}
	nextSize := w.n + int64(len(p))
	if nextSize == w.size {
		// last write. check hash.
		sum := w.h.Sum(nil)
		if !bytes.Equal(sum, w.d.sum[:]) {
			return 0, w.seterr(fmt.Errorf("file content changed underfoot"))
		}
		if w.testHookBeforeFinalWrite != nil {
			w.testHookBeforeFinalWrite(w.f)
		}
	}
	if nextSize > w.size {
		return 0, w.seterr(fmt.Errorf("content exceeds expected size: %d > %d", nextSize, w.size))
	}
	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, w.seterr(err)
}

// copyNamedFile copies file into name, expecting it to have the given Digest
// and size, if that file is not present already.
func (c *DiskCache) copyNamedFile(name string, file io.Reader, out Digest, size int64) error {
	info, err := os.Stat(name)
	if err == nil && info.Size() == size {
		// File already exists with correct size. This is good enough.
		// We can skip expensive hash checks.
		//
		// TODO: Do the hash check, but give caller a way to skip it.
		return nil
	}

	// Copy file to cache directory.
	mode := os.O_RDWR | os.O_CREATE
	if err == nil && info.Size() > size { // shouldn't happen but fix in case
		mode |= os.O_TRUNC
	}
	f, err := os.OpenFile(name, mode, 0o666)
	if err != nil {
		return err
	}
	defer f.Close()
	if size == 0 {
		// File now exists with correct size.
		// Only one possible zero-length file, so contents are OK too.
		// Early return here makes sure there's a "last byte" for code below.
		return nil
	}

	// From here on, if any of the I/O writing the file fails,
	// we make a best-effort attempt to truncate the file f
	// before returning, to avoid leaving bad bytes in the file.

	// Copy file to f, but also into h to double-check hash.
	cw := &checkWriter{
		d:    out,
		size: size,
		h:    sha256.New(),
		f:    f,
		w:    f,

		testHookBeforeFinalWrite: c.testHookBeforeFinalWrite,
	}
	n, err := io.Copy(cw, file)
	if err != nil {
		f.Truncate(0)
		return err
	}
	if n < size {
		f.Truncate(0)
		return io.ErrUnexpectedEOF
	}

	if err := f.Close(); err != nil {
		// Data might not have been written,
		// but file may look like it is the right size.
		// To be extra careful, remove cached file.
		os.Remove(name)
		return err
	}
	os.Chtimes(name, c.now(), c.now()) // mainly for tests

	return nil
}
