// Package blob implements a content-addressable disk cache for blobs and
// manifests.
//
// Modul: cache_resolve.go - Resolve, Link, Unlink und Pfad-Funktionen
// Enthaelt: Resolve, Link, Unlink, Links, manifestPath und Hilfsfunktionen
package blob

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/server/internal/internal/names"
)

func readAndSum(filename string, limit int64) (data []byte, _ Digest, err error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, Digest{}, err
	}
	defer f.Close()

	h := sha256.New()
	r := io.TeeReader(f, h)
	data, err = io.ReadAll(io.LimitReader(r, limit))
	if err != nil {
		return nil, Digest{}, err
	}
	var d Digest
	h.Sum(d.sum[:0])
	return data, d, nil
}

//lint:ignore U1000 used for debugging purposes as needed in tests
var debug = false

// debugger returns a function that can be used to add a step to the error message.
// The error message will be a list of steps that were taken before the error occurred.
// The steps are added in the order they are called.
//
// To set the error message, call the returned function with an empty string.
//
//lint:ignore U1000 used for debugging purposes as needed in tests
func debugger(err *error) func(step string) {
	if !debug {
		return func(string) {}
	}
	var steps []string
	return func(step string) {
		if step == "" && *err != nil {
			*err = fmt.Errorf("%q: %w", steps, *err)
			return
		}
		steps = append(steps, step)
		if len(steps) > 100 {
			// shift hints in case of a bug that causes a lot of hints
			copy(steps, steps[1:])
			steps = steps[:100]
		}
	}
}

// Resolve resolves a name to a digest. The name is expected to
// be in either of the following forms:
//
//	@<digest>
//	<name>@<digest>
//	<name>
//
// If a digest is provided, it is returned as is and nothing else happens.
//
// If a name is provided for a manifest that exists in the cache, the digest
// of the manifest is returned. If there is no manifest in the cache, it
// returns [fs.ErrNotExist].
//
// To cover the case where a manifest may change without the cache knowing
// (e.g. it was reformatted or modified by hand), the manifest data read and
// hashed is passed to a PutBytes call to ensure that the manifest is in the
// blob store. This is done to ensure that future calls to [Get] succeed in
// these cases.
func (c *DiskCache) Resolve(name string) (Digest, error) {
	name, digest := splitNameDigest(name)
	if digest != "" {
		return ParseDigest(digest)
	}

	// We want to address manifests files by digest using Get. That requires
	// them to be blobs. This cannot be directly accomplished by looking in
	// the blob store because manifests can change without Ollama knowing
	// (e.g. a user modifies a manifests by hand then pushes it to update
	// their model). We also need to support the blob caches inherited from
	// older versions of Ollama, which do not store manifests in the blob
	// store, so for these cases, we need to handle adding the manifests to
	// the blob store, just in time.
	//
	// So now we read the manifests file, hash it, and copy it to the blob
	// store if it's not already there.
	//
	// This should be cheap because manifests are small, and accessed
	// infrequently.
	file, err := c.manifestPath(name)
	if err != nil {
		return Digest{}, err
	}

	data, d, err := readAndSum(file, 1<<20)
	if err != nil {
		return Digest{}, err
	}

	// Ideally we'd read the "manifest" file as a manifest to the blob file,
	// but we are not changing this yet, so copy the manifest to the blob
	// store so it can be addressed by digest subsequent calls to Get.
	if err := PutBytes(c, d, data); err != nil {
		return Digest{}, err
	}
	return d, nil
}

// Link creates a symbolic reference in the cache that maps the provided name
// to a blob identified by its digest, making it retrievable by name using
// [Resolve].
//
// It returns an error if either the name or digest is invalid, or if link
// creation encounters any issues.
func (c *DiskCache) Link(name string, d Digest) error {
	manifest, err := c.manifestPath(name)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(c.GetFile(d), os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()

	// TODO(bmizerany): test this happens only if the blob was found to
	// avoid leaving debris
	if err := os.MkdirAll(filepath.Dir(manifest), 0o777); err != nil {
		return err
	}

	info, err := f.Stat()
	if err != nil {
		return err
	}

	// Copy manifest to cache directory.
	return c.copyNamedFile(manifest, f, d, info.Size())
}

// Unlink unlinks the manifest by name from the cache. If the name is not
// found. If a manifest is removed ok will be true, otherwise false. If an
// error occurs, it returns ok false, and the error.
func (c *DiskCache) Unlink(name string) (ok bool, _ error) {
	manifest, err := c.manifestPath(name)
	if err != nil {
		return false, err
	}
	err = os.Remove(manifest)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return true, err
}

// Links returns a sequence of link names. The sequence is in lexical order.
// Names are converted from their relative path form to their name form but are
// not guaranteed to be valid. Callers should validate the names before using.
func (c *DiskCache) Links() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for path, err := range c.links() {
			if err != nil {
				yield("", err)
				return
			}
			if !yield(pathToName(path), nil) {
				return
			}
		}
	}
}

// pathToName converts a path to a name. It is the inverse of nameToPath. The
// path is assumed to be in filepath.ToSlash format.
func pathToName(s string) string {
	s = strings.TrimPrefix(s, "manifests/")
	rr := []rune(s)
	for i := len(rr) - 1; i > 0; i-- {
		if rr[i] == '/' {
			rr[i] = ':'
			return string(rr)
		}
	}
	return s
}

// manifestPath finds the first manifest file on disk that matches the given
// name using a case-insensitive comparison. If no manifest file is found, it
// returns the path where the manifest file would be if it existed.
//
// If two manifest files exists on disk that match the given name using a
// case-insensitive comparison, the one that sorts first, lexically, is
// returned.
func (c *DiskCache) manifestPath(name string) (string, error) {
	np, err := nameToPath(name)
	if err != nil {
		return "", err
	}

	maybe := filepath.Join("manifests", np)
	for l, err := range c.links() {
		if err != nil {
			return "", err
		}
		if strings.EqualFold(maybe, l) {
			return filepath.Join(c.dir, l), nil
		}
	}
	return filepath.Join(c.dir, maybe), nil
}

// links returns a sequence of links in the cache in lexical order.
func (c *DiskCache) links() iter.Seq2[string, error] {
	// TODO(bmizerany): reuse empty dirnames if exist
	return func(yield func(string, error) bool) {
		fsys := os.DirFS(c.dir)
		manifests, err := fs.Glob(fsys, "manifests/*/*/*/*")
		if err != nil {
			yield("", err)
			return
		}
		for _, manifest := range manifests {
			if !yield(manifest, nil) {
				return
			}
		}
	}
}

var errInvalidName = errors.New("invalid name")

func nameToPath(name string) (_ string, err error) {
	n := names.Parse(name)
	if !n.IsFullyQualified() {
		return "", errInvalidName
	}
	return filepath.Join(n.Host(), n.Namespace(), n.Model(), n.Tag()), nil
}
