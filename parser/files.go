// Package parser - Modelfile-Parser für Ollama
// Modul files: Dateisystem-Operationen und Pfadverarbeitung
package parser

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
)

// isNotExist prüft ob der Fehler ein os.ErrNotExist ist
func isNotExist(err error) bool {
	return errors.Is(err, os.ErrNotExist)
}

func fileDigestMap(path string) (map[string]string, error) {
	fl := make(map[string]string)

	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	var files []string
	if fi.IsDir() {
		fs, err := filesForModel(path)
		if err != nil {
			return nil, err
		}

		for _, f := range fs {
			f, err := filepath.EvalSymlinks(f)
			if err != nil {
				return nil, err
			}

			rel, err := filepath.Rel(path, f)
			if err != nil {
				return nil, err
			}

			if !filepath.IsLocal(rel) {
				return nil, fmt.Errorf("insecure path: %s", rel)
			}

			files = append(files, f)
		}
	} else {
		files = []string{path}
	}

	var mu sync.Mutex
	var g errgroup.Group
	g.SetLimit(max(runtime.GOMAXPROCS(0)-1, 1))
	for _, f := range files {
		g.Go(func() error {
			digest, err := digestForFile(f)
			if err != nil {
				return err
			}

			mu.Lock()
			defer mu.Unlock()
			fl[f] = digest
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return fl, nil
}

func digestForFile(filename string) (string, error) {
	filepath, err := filepath.EvalSymlinks(filename)
	if err != nil {
		return "", err
	}

	bin, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, bin); err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", hash.Sum(nil)), nil
}

func filesForModel(path string) ([]string, error) {
	detectContentType := func(path string) (string, error) {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		defer f.Close()

		var b bytes.Buffer
		b.Grow(512)

		if _, err := io.CopyN(&b, f, 512); err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}

		contentType, _, _ := strings.Cut(http.DetectContentType(b.Bytes()), ";")
		return contentType, nil
	}

	glob := func(pattern, contentType string) ([]string, error) {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, err
		}

		for _, match := range matches {
			if ct, err := detectContentType(match); err != nil {
				return nil, err
			} else if len(contentType) > 0 && ct != contentType {
				return nil, fmt.Errorf("invalid content type: expected %s for %s", ct, match)
			}
		}

		return matches, nil
	}

	var files []string
	// some safetensors files do not properly match "application/octet-stream", so skip checking their contentType
	if st, _ := glob(filepath.Join(path, "model*.safetensors"), ""); len(st) > 0 {
		// safetensors files might be unresolved git lfs references; skip if they are
		// covers model-x-of-y.safetensors, model.fp32-x-of-y.safetensors, model.safetensors
		files = append(files, st...)
	} else if st, _ := glob(filepath.Join(path, "consolidated*.safetensors"), ""); len(st) > 0 {
		// covers consolidated.safetensors
		files = append(files, st...)
	} else if pt, _ := glob(filepath.Join(path, "pytorch_model*.bin"), "application/zip"); len(pt) > 0 {
		// pytorch files might also be unresolved git lfs references; skip if they are
		// covers pytorch_model-x-of-y.bin, pytorch_model.fp32-x-of-y.bin, pytorch_model.bin
		files = append(files, pt...)
	} else if pt, _ := glob(filepath.Join(path, "consolidated*.pth"), "application/zip"); len(pt) > 0 {
		// pytorch files might also be unresolved git lfs references; skip if they are
		// covers consolidated.x.pth, consolidated.pth
		files = append(files, pt...)
	} else if gg, _ := glob(filepath.Join(path, "*.gguf"), "application/octet-stream"); len(gg) > 0 {
		// covers gguf files ending in .gguf
		files = append(files, gg...)
	} else if gg, _ := glob(filepath.Join(path, "*.bin"), "application/octet-stream"); len(gg) > 0 {
		// covers gguf files ending in .bin
		files = append(files, gg...)
	} else {
		return nil, ErrModelNotFound
	}

	// add configuration files, json files are detected as text/plain
	js, err := glob(filepath.Join(path, "*.json"), "text/plain")
	if err != nil {
		return nil, err
	}
	files = append(files, js...)

	// bert models require a nested config.json
	// TODO(mxyng): merge this with the glob above
	js, err = glob(filepath.Join(path, "**/*.json"), "text/plain")
	if err != nil {
		return nil, err
	}
	files = append(files, js...)

	// add tokenizer.model if it exists (tokenizer.json is automatically picked up by the previous glob)
	// tokenizer.model might be a unresolved git lfs reference; error if it is
	if tks, _ := glob(filepath.Join(path, "tokenizer.model"), "application/octet-stream"); len(tks) > 0 {
		files = append(files, tks...)
	} else if tks, _ := glob(filepath.Join(path, "**/tokenizer.model"), "text/plain"); len(tks) > 0 {
		// some times tokenizer.model is in a subdirectory (e.g. meta-llama/Meta-Llama-3-8B)
		files = append(files, tks...)
	}

	return files, nil
}

func expandPathImpl(path, relativeDir string, currentUserFunc func() (*user.User, error), lookupUserFunc func(string) (*user.User, error)) (string, error) {
	if filepath.IsAbs(path) || strings.HasPrefix(path, "\\") || strings.HasPrefix(path, "/") {
		return filepath.Abs(path)
	} else if strings.HasPrefix(path, "~") {
		var homeDir string

		if path == "~" || strings.HasPrefix(path, "~/") {
			// Current user's home directory
			currentUser, err := currentUserFunc()
			if err != nil {
				return "", fmt.Errorf("failed to get current user: %w", err)
			}
			homeDir = currentUser.HomeDir
			path = strings.TrimPrefix(path, "~")
		} else {
			// Specific user's home directory
			parts := strings.SplitN(path[1:], "/", 2)
			userInfo, err := lookupUserFunc(parts[0])
			if err != nil {
				return "", fmt.Errorf("failed to find user '%s': %w", parts[0], err)
			}
			homeDir = userInfo.HomeDir
			if len(parts) > 1 {
				path = "/" + parts[1]
			} else {
				path = ""
			}
		}

		path = filepath.Join(homeDir, path)
	} else {
		path = filepath.Join(relativeDir, path)
	}

	return filepath.Abs(path)
}

func expandPath(path, relativeDir string) (string, error) {
	return expandPathImpl(path, relativeDir, user.Current, user.Lookup)
}
