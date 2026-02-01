// create_convert_safetensors.go - Safetensors-Konvertierung
//
// Enthaelt:
// - convertModelFromFiles: Dispatcher fuer Konvertierung
// - detectModelTypeFromFiles: Dateityp-Erkennung
// - convertFromSafetensors: Safetensors zu GGUF
// - kvFromLayers: KV-Config Extraktion
package server

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
)

// convertModelFromFiles konvertiert Dateien basierend auf erkanntem Typ
func convertModelFromFiles(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	switch detectModelTypeFromFiles(files) {
	case "safetensors":
		layers, err := convertFromSafetensors(files, baseLayers, isAdapter, fn)
		if err != nil {
			slog.Error("error converting from safetensors", "error", err)
			return nil, err
		}
		return layers, nil
	case "gguf":
		if len(files) == 0 {
			return nil, errNoFilesProvided
		} else if len(files) > 1 && isAdapter {
			return nil, errOnlyOneAdapterSupported
		}

		var digest string
		var allLayers []*layerGGML
		for _, v := range files {
			digest = v
			layers, err := ggufLayers(digest, fn)
			if err != nil {
				return nil, err
			}
			allLayers = append(allLayers, layers...)
		}
		return allLayers, nil
	default:
		return nil, errUnknownType
	}
}

// detectModelTypeFromFiles erkennt den Dateityp anhand der Dateiendung oder Magic Bytes
func detectModelTypeFromFiles(files map[string]string) string {
	for fn := range files {
		if strings.HasSuffix(fn, ".safetensors") {
			return "safetensors"
		} else if strings.HasSuffix(fn, ".gguf") {
			return "gguf"
		} else {
			// Versuche GGUF anhand der Magic Bytes zu erkennen
			blobPath, err := manifest.BlobsPath(files[fn])
			if err != nil {
				slog.Error("error getting blobs path", "file", fn)
				return ""
			}

			f, err := os.Open(blobPath)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}
			defer f.Close()

			buf := make([]byte, 4)
			_, err = f.Read(buf)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}

			ct := ggml.DetectContentType(buf)
			if ct == "gguf" {
				return "gguf"
			}
		}
	}

	return ""
}

// convertFromSafetensors konvertiert Safetensors-Dateien zu GGUF
func convertFromSafetensors(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	tmpDir, err := os.MkdirTemp(envconfig.Models(), "ollama-safetensors")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	// Root-Verzeichnis fuer Pfadvalidierung
	root, err := os.OpenRoot(tmpDir)
	if err != nil {
		return nil, err
	}
	defer root.Close()

	for fp, digest := range files {
		if !fs.ValidPath(fp) {
			return nil, fmt.Errorf("%w: %s", errFilePath, fp)
		}
		if _, err := root.Stat(fp); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s: %s", errFilePath, err, fp)
		}

		blobPath, err := manifest.BlobsPath(digest)
		if err != nil {
			return nil, err
		}
		if err := createLink(blobPath, filepath.Join(tmpDir, fp)); err != nil {
			return nil, err
		}
	}

	t, err := os.CreateTemp(tmpDir, "fp16")
	if err != nil {
		return nil, err
	}
	defer t.Close()

	var mediaType string
	if !isAdapter {
		fn(api.ProgressResponse{Status: "converting model"})
		mediaType = "application/vnd.ollama.image.model"
		if err := convert.ConvertModel(os.DirFS(tmpDir), t); err != nil {
			return nil, err
		}
	} else {
		kv, err := kvFromLayers(baseLayers)
		if err != nil {
			return nil, err
		}
		fn(api.ProgressResponse{Status: "converting adapter"})
		mediaType = "application/vnd.ollama.image.adapter"
		if err := convert.ConvertAdapter(os.DirFS(tmpDir), t, kv); err != nil {
			return nil, err
		}
	}

	if _, err := t.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(t, mediaType)
	if err != nil {
		return nil, err
	}

	bin, err := layer.Open()
	if err != nil {
		return nil, err
	}
	defer bin.Close()

	f, err := ggml.Decode(bin, -1)
	if err != nil {
		return nil, err
	}
	layers := []*layerGGML{{layer, f}}

	if !isAdapter {
		return detectChatTemplate(layers)
	}
	return layers, nil
}

// kvFromLayers extrahiert KV-Config aus Base-Layern
func kvFromLayers(baseLayers []*layerGGML) (ofs.Config, error) {
	for _, l := range baseLayers {
		if l.GGML != nil {
			return l.KV(), nil
		}
	}
	return ggml.KV{}, fmt.Errorf("no base model was found")
}
