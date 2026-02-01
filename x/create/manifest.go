// Modul: manifest.go
// Beschreibung: Funktionen zum Laden und Auflösen von Manifest-Dateien.
// Enthält: defaultManifestDir, defaultBlobDir, resolveManifestPath, loadManifest, loadModelConfig.

package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// defaultManifestDir returns the manifest storage directory.
func defaultManifestDir() string {
	return filepath.Join(envconfig.Models(), "manifests")
}

// defaultBlobDir returns the blob storage directory.
func defaultBlobDir() string {
	return filepath.Join(envconfig.Models(), "blobs")
}

// resolveManifestPath converts a model name to a manifest file path.
func resolveManifestPath(modelName string) string {
	host := "registry.ollama.ai"
	namespace := "library"
	name := modelName
	tag := "latest"

	if idx := strings.LastIndex(name, ":"); idx != -1 {
		tag = name[idx+1:]
		name = name[:idx]
	}

	parts := strings.Split(name, "/")
	switch len(parts) {
	case 3:
		host = parts[0]
		namespace = parts[1]
		name = parts[2]
	case 2:
		namespace = parts[0]
		name = parts[1]
	}

	return filepath.Join(defaultManifestDir(), host, namespace, name, tag)
}

// loadManifest loads a manifest for the given model name.
func loadManifest(modelName string) (*Manifest, error) {
	manifestPath := resolveManifestPath(modelName)

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, err
	}

	return &manifest, nil
}

// loadModelConfig loads the config blob for a model.
func loadModelConfig(modelName string) (*ModelConfig, error) {
	manifest, err := loadManifest(modelName)
	if err != nil {
		return nil, err
	}

	// Read the config blob
	blobName := strings.Replace(manifest.Config.Digest, ":", "-", 1)
	blobPath := filepath.Join(defaultBlobDir(), blobName)

	data, err := os.ReadFile(blobPath)
	if err != nil {
		return nil, err
	}

	var config ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}
