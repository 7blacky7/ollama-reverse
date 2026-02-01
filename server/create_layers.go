// create_layers.go - Layer-Management fuer Model-Erstellung
//
// Enthaelt:
// - createModel: Erstellt ein Model aus Layern
// - removeLayer: Entfernt Layer nach MediaType
// - setTemplate: Setzt Template-Layer
// - setSystem: Setzt System-Layer
// - setLicense: Setzt Lizenz-Layer
// - setParameters: Setzt Parameter-Layer
// - setMessages: Setzt Messages-Layer
// - createConfigLayer: Erstellt Config-Layer
// - createLink/copyFile: Hilfsfunktionen fuer Dateiverlinkung
package server

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// createModel erstellt ein Model aus den gegebenen Layern
func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, config *model.ConfigV2, fn func(resp api.ProgressResponse)) (err error) {
	var layers []manifest.Layer

	for _, layer := range baseLayers {
		if layer.GGML != nil {
			quantType := strings.ToUpper(cmp.Or(r.Quantize, r.Quantization))
			if quantType != "" && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.model" {
				want, err := ggml.ParseFileType(quantType)
				if err != nil {
					return err
				}

				ft := layer.GGML.KV().FileType()
				if !slices.Contains([]string{"F16", "F32"}, ft.String()) {
					return errors.New("quantization is only supported for F16 and F32 models")
				} else if ft != want {
					layer, err = quantizeLayer(layer, quantType, fn)
					if err != nil {
						return err
					}
				}
			}
			config.ModelFormat = cmp.Or(config.ModelFormat, layer.GGML.Name())
			config.ModelFamily = cmp.Or(config.ModelFamily, layer.GGML.KV().Architecture())
			config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(layer.GGML.KV().ParameterCount()))
			config.FileType = cmp.Or(config.FileType, layer.GGML.KV().FileType().String())
			config.ModelFamilies = append(config.ModelFamilies, layer.GGML.KV().Architecture())
		}
		layers = append(layers, layer.Layer)
	}

	if r.Template != "" {
		layers, err = setTemplate(layers, r.Template)
		if err != nil {
			return err
		}
	}

	if r.System != "" {
		layers, err = setSystem(layers, r.System)
		if err != nil {
			return err
		}
	}

	if r.License != nil {
		switch l := r.License.(type) {
		case string:
			if l != "" {
				layers, err = setLicense(layers, l)
				if err != nil {
					return err
				}
			}
		case any:
			var licenses []string
			b, _ := json.Marshal(l)
			if err := json.Unmarshal(b, &licenses); err != nil {
				return err
			}
			for _, v := range licenses {
				layers, err = setLicense(layers, v)
				if err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unknown license type: %T", l)
		}
	}

	layers, err = setParameters(layers, r.Parameters)
	if err != nil {
		return err
	}

	layers, err = setMessages(layers, r.Messages)
	if err != nil {
		return err
	}

	configLayer, err := createConfigLayer(layers, *config)
	if err != nil {
		return err
	}

	for _, layer := range layers {
		if layer.Status != "" {
			fn(api.ProgressResponse{Status: layer.Status})
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		return err
	}

	return nil
}

// removeLayer entfernt alle Layer mit dem angegebenen MediaType
func removeLayer(layers []manifest.Layer, mediatype string) []manifest.Layer {
	return slices.DeleteFunc(layers, func(layer manifest.Layer) bool {
		if layer.MediaType != mediatype {
			return false
		}

		if err := layer.Remove(); err != nil {
			slog.Warn("couldn't remove blob", "digest", layer.Digest, "error", err)
			return true
		}

		return true
	})
}

// setTemplate setzt den Template-Layer
func setTemplate(layers []manifest.Layer, t string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.template")

	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}
	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}

	blob := strings.NewReader(t)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.template")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

// setSystem setzt den System-Layer
func setSystem(layers []manifest.Layer, s string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.system")

	if s != "" {
		blob := strings.NewReader(s)
		layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.system")
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// setLicense fuegt einen Lizenz-Layer hinzu
func setLicense(layers []manifest.Layer, l string) ([]manifest.Layer, error) {
	blob := strings.NewReader(l)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.license")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

// setParameters setzt den Parameters-Layer
func setParameters(layers []manifest.Layer, p map[string]any) ([]manifest.Layer, error) {
	if p == nil {
		p = make(map[string]any)
	}

	// Existierende Parameter laden und mergen
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}

		digestPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		fn, err := os.Open(digestPath)
		if err != nil {
			return nil, err
		}
		defer fn.Close()

		var existing map[string]any
		if err := json.NewDecoder(fn).Decode(&existing); err != nil {
			return nil, err
		}

		for k, v := range existing {
			if _, exists := p[k]; exists {
				continue
			}
			p[k] = v
		}
	}

	if len(p) == 0 {
		return layers, nil
	}

	layers = removeLayer(layers, "application/vnd.ollama.image.params")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(p); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

// setMessages setzt den Messages-Layer
func setMessages(layers []manifest.Layer, m []api.Message) ([]manifest.Layer, error) {
	// Alte Messages beibehalten wenn keine neuen angegeben
	if len(m) == 0 {
		return layers, nil
	}

	fmt.Printf("removing old messages\n")
	layers = removeLayer(layers, "application/vnd.ollama.image.messages")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(m); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.messages")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

// createConfigLayer erstellt den Config-Layer mit RootFS-Info
func createConfigLayer(layers []manifest.Layer, config model.ConfigV2) (*manifest.Layer, error) {
	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}
	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, err
	}

	return &layer, nil
}

// createLink erstellt einen Symlink oder kopiert die Datei als Fallback
func createLink(src, dst string) error {
	// Subdirs fuer dst erstellen
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	_ = os.Remove(dst)
	if err := os.Symlink(src, dst); err != nil {
		if err := copyFile(src, dst); err != nil {
			return err
		}
	}
	return nil
}

// copyFile kopiert eine Datei
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
