// create_layers.go - Haupt-Layer-Erstellung fuer Models
//
// Dieses Modul enthaelt:
// - createModel: Erstellt ein Model aus Layern mit Quantisierung
// - removeLayer: Entfernt Layer nach MediaType
//
// Weitere Layer-Funktionen sind ausgelagert:
// - layer_setters.go: setTemplate, setSystem, setLicense, setParameters, setMessages
// - layer_utils.go: createConfigLayer, createLink, copyFile
package server

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// createModel erstellt ein Model aus den gegebenen Layern
// Verarbeitet Quantisierung, setzt Config-Werte und schreibt das Manifest
func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, config *model.ConfigV2, fn func(resp api.ProgressResponse)) (err error) {
	var layers []manifest.Layer

	// Basis-Layer verarbeiten
	for _, layer := range baseLayers {
		if layer.GGML != nil {
			// Quantisierung pruefen und anwenden
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

			// Config-Werte aus GGML setzen
			config.ModelFormat = cmp.Or(config.ModelFormat, layer.GGML.Name())
			config.ModelFamily = cmp.Or(config.ModelFamily, layer.GGML.KV().Architecture())
			config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(layer.GGML.KV().ParameterCount()))
			config.FileType = cmp.Or(config.FileType, layer.GGML.KV().FileType().String())
			config.ModelFamilies = append(config.ModelFamilies, layer.GGML.KV().Architecture())
		}
		layers = append(layers, layer.Layer)
	}

	// Template setzen
	if r.Template != "" {
		layers, err = setTemplate(layers, r.Template)
		if err != nil {
			return err
		}
	}

	// System-Prompt setzen
	if r.System != "" {
		layers, err = setSystem(layers, r.System)
		if err != nil {
			return err
		}
	}

	// Lizenzen setzen
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

	// Parameter setzen
	layers, err = setParameters(layers, r.Parameters)
	if err != nil {
		return err
	}

	// Messages setzen
	layers, err = setMessages(layers, r.Messages)
	if err != nil {
		return err
	}

	// Config-Layer erstellen
	configLayer, err := createConfigLayer(layers, *config)
	if err != nil {
		return err
	}

	// Status-Updates senden
	for _, layer := range layers {
		if layer.Status != "" {
			fn(api.ProgressResponse{Status: layer.Status})
		}
	}

	// Manifest schreiben
	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		return err
	}

	return nil
}

// removeLayer entfernt alle Layer mit dem angegebenen MediaType
// Gibt die bereinigte Layer-Liste zurueck
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
