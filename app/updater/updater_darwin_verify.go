// Package updater - Verifizierungs-Funktionen fuer macOS.
// Dieses Modul enthaelt die Download-Verifizierung und zugehoerige Hilfsfunktionen.

package updater

import (
	"archive/zip"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// verifyDownload verifiziert das heruntergeladene Update-Bundle.
func verifyDownload() error {
	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}
	slog.Debug("verifying update", "bundle", bundle)

	// Extract zip file into a temporary location so we can run the cert verification routines
	dir, err := os.MkdirTemp("", "ollama_update_verify")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	r, err := zip.OpenReader(bundle)
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer r.Close()

	// Extract directories and regular files
	links := []*zip.File{}
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, "/") {
			d := filepath.Join(dir, f.Name)
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}
		src, err := f.Open()
		if err != nil {
			return fmt.Errorf("failed to open bundle file %s: %w", f.Name, err)
		}
		destName := filepath.Join(dir, f.Name)
		// Verify directory first
		d := filepath.Dir(destName)
		if _, err := os.Stat(d); err != nil {
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
		}
		destFile, err := os.OpenFile(destName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			return fmt.Errorf("failed to open output file %s: %w", destName, err)
		}
		defer destFile.Close()
		if _, err := io.Copy(destFile, src); err != nil {
			return fmt.Errorf("failed to open extract file %s: %w", destName, err)
		}
	}

	// Process symlinks
	for _, f := range links {
		src, err := f.Open()
		if err != nil {
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			return err
		}
		link := string(buf)
		if link[0] == '/' {
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if strings.HasPrefix(filepath.Join(filepath.Dir(f.Name), link), "..") {
			return fmt.Errorf("bundle contains link outside of contents %s -> %s", f.Name, link)
		}
		if err = os.Symlink(link, filepath.Join(dir, f.Name)); err != nil {
			return err
		}
	}

	if err := verifyExtractedBundle(filepath.Join(dir, "Ollama.app")); err != nil {
		return fmt.Errorf("signature verification failed: %s", err)
	}
	return nil
}
