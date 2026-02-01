// layer_utils.go - Utility-Funktionen fuer Layer-Management
//
// Dieses Modul enthaelt:
// - createConfigLayer: Erstellt den Config-Layer mit RootFS-Info
// - createLink: Erstellt Symlink oder kopiert als Fallback
// - copyFile: Kopiert eine Datei
package server

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// createConfigLayer erstellt den Config-Layer mit RootFS-Info
// Sammelt alle Layer-Digests fuer die DiffIDs
func createConfigLayer(layers []manifest.Layer, config model.ConfigV2) (*manifest.Layer, error) {
	// Digests sammeln
	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}
	config.RootFS.DiffIDs = digests

	// Config als JSON serialisieren
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return nil, err
	}

	// Layer erstellen
	layer, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, err
	}

	return &layer, nil
}

// createLink erstellt einen Symlink oder kopiert die Datei als Fallback
// Erstellt automatisch fehlende Verzeichnisse fuer dst
func createLink(src, dst string) error {
	// Subdirs fuer dst erstellen
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	// Bestehende Datei/Link entfernen
	_ = os.Remove(dst)

	// Symlink versuchen, bei Fehler kopieren
	if err := os.Symlink(src, dst); err != nil {
		if err := copyFile(src, dst); err != nil {
			return err
		}
	}
	return nil
}

// copyFile kopiert eine Datei von src nach dst
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}
