// Package server - Model-Operationen (Laden, Kopieren, Bereinigen)
//
// Diese Datei enthält:
// - GetModel: Model aus Manifest laden
// - CopyModel: Model kopieren
// - deleteUnusedLayers: Unbenutzte Layer löschen
// - PruneLayers: Verwaiste Blobs bereinigen
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// GetModel lädt ein Model anhand seines Namens aus dem lokalen Manifest
func GetModel(name string) (*Model, error) {
	n := model.ParseName(name)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Name:      n.String(),
		ShortName: n.DisplayShortest(),
		Digest:    mf.Digest(),
		Template:  template.DefaultTemplate,
	}

	if mf.Config.Digest != "" {
		filename, err := manifest.BlobsPath(mf.Config.Digest)
		if err != nil {
			return nil, err
		}

		configFile, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer configFile.Close()

		if err := json.NewDecoder(configFile).Decode(&m.Config); err != nil {
			return nil, err
		}
	}

	for _, layer := range mf.Layers {
		filename, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			m.ModelPath = filename
			m.ParentModel = layer.From
		case "application/vnd.ollama.image.embed":
			// Veraltet seit Version > 0.1.2
			// TODO: Diese Warnung in einer zukünftigen Version entfernen
			slog.Info("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			m.AdapterPaths = append(m.AdapterPaths, filename)
		case "application/vnd.ollama.image.projector":
			m.ProjectorPaths = append(m.ProjectorPaths, filename)
		case "application/vnd.ollama.image.prompt",
			"application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.Template, err = template.Parse(string(bts))
			if err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.System = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// Model-Optionen als Map parsen für explizite Feldprüfung
			if err = json.NewDecoder(params).Decode(&m.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.messages":
			msgs, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer msgs.Close()

			if err = json.NewDecoder(msgs).Decode(&m.Messages); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			m.License = append(m.License, string(bts))
		}
	}

	return m, nil
}

// CopyModel kopiert ein Model von src nach dst
func CopyModel(src, dst model.Name) error {
	if !dst.IsFullyQualified() {
		return model.Unqualified(dst)
	}
	if !src.IsFullyQualified() {
		return model.Unqualified(src)
	}

	if src.Filepath() == dst.Filepath() {
		return nil
	}

	manifests, err := manifest.Path()
	if err != nil {
		return err
	}

	dstpath := filepath.Join(manifests, dst.Filepath())
	if err := os.MkdirAll(filepath.Dir(dstpath), 0o755); err != nil {
		return err
	}

	srcpath := filepath.Join(manifests, src.Filepath())
	srcfile, err := os.Open(srcpath)
	if err != nil {
		return err
	}
	defer srcfile.Close()

	dstfile, err := os.Create(dstpath)
	if err != nil {
		return err
	}
	defer dstfile.Close()

	_, err = io.Copy(dstfile, srcfile)
	return err
}

// deleteUnusedLayers löscht Layer die nicht mehr von Manifests referenziert werden
func deleteUnusedLayers(deleteMap map[string]struct{}) error {
	// Korrupte Manifests ignorieren um Löschung verwaister Layer nicht zu blockieren
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	for _, manifest := range manifests {
		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
	}

	// Nur Dateien löschen die noch in deleteMap sind
	for k := range deleteMap {
		fp, err := manifest.BlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if err := os.Remove(fp); err != nil {
			slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
			continue
		}
	}

	return nil
}

// PruneLayers entfernt verwaiste Blobs die von keinem Manifest mehr referenziert werden
func PruneLayers() error {
	deleteMap := make(map[string]struct{})
	p, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't read dir '%s': %v", p, err))
		return err
	}

	for _, blob := range blobs {
		name := blob.Name()
		name = strings.ReplaceAll(name, "-", ":")

		_, err := manifest.BlobsPath(name)
		if err != nil {
			if errors.Is(err, manifest.ErrInvalidDigestFormat) {
				// Ungültige Blobs entfernen (z.B. unvollständige Downloads)
				if err := os.Remove(filepath.Join(p, blob.Name())); err != nil {
					slog.Error("couldn't remove blob", "blob", blob.Name(), "error", err)
				}
			}

			continue
		}

		deleteMap[name] = struct{}{}
	}

	slog.Info(fmt.Sprintf("total blobs: %d", len(deleteMap)))

	if err := deleteUnusedLayers(deleteMap); err != nil {
		slog.Error(fmt.Sprintf("couldn't remove unused layers: %v", err))
		return nil
	}

	slog.Info(fmt.Sprintf("total unused blobs removed: %d", len(deleteMap)))

	return nil
}
