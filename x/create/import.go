// Modul: import.go
// Beschreibung: Hauptfunktion zum Importieren von Safetensors-Modellen.
// Enthält: CreateSafetensorsModel.

package create

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// CreateSafetensorsModel imports a standard safetensors model from a directory.
// This handles Hugging Face style models with config.json and *.safetensors files.
// Stores each tensor as a separate blob for fine-grained deduplication.
// If quantize is non-empty (e.g., "fp8"), eligible tensors will be quantized.
func CreateSafetensorsModel(modelName, modelDir, quantize string, createLayer LayerCreator, createTensorLayer QuantizingTensorLayerCreator, writeManifest ManifestWriter, fn func(status string)) error {
	var layers []LayerInfo
	var configLayer LayerInfo

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	// Process all safetensors files
	if err := processSafetensorsFiles(entries, modelDir, quantize, createTensorLayer, &layers, fn); err != nil {
		return err
	}

	// Process all JSON config files
	if err := processConfigFiles(entries, modelDir, createLayer, &layers, &configLayer, fn); err != nil {
		return err
	}

	if configLayer.Digest == "" {
		return fmt.Errorf("config.json not found in %s", modelDir)
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))

	if err := writeManifest(modelName, configLayer, layers); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}

// processSafetensorsFiles processes all .safetensors files in the directory
func processSafetensorsFiles(entries []os.DirEntry, modelDir, quantize string, createTensorLayer QuantizingTensorLayerCreator, layers *[]LayerInfo, fn func(status string)) error {
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}

		stPath := filepath.Join(modelDir, entry.Name())

		extractor, err := safetensors.OpenForExtraction(stPath)
		if err != nil {
			return fmt.Errorf("failed to open %s: %w", stPath, err)
		}

		tensorNames := extractor.ListTensors()
		quantizeMsg := ""
		if quantize != "" {
			quantizeMsg = fmt.Sprintf(", quantizing to %s", quantize)
		}
		fn(fmt.Sprintf("importing %s (%d tensors%s)", entry.Name(), len(tensorNames), quantizeMsg))

		for _, tensorName := range tensorNames {
			td, err := extractor.GetTensor(tensorName)
			if err != nil {
				extractor.Close()
				return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
			}

			// Determine quantization type for this tensor (empty string if not quantizing)
			quantizeType := ""
			if quantize != "" && ShouldQuantizeTensor(tensorName, td.Shape) {
				quantizeType = quantize
			}

			// Store as minimal safetensors format (88 bytes header overhead)
			// This enables native mmap loading via mlx_load_safetensors
			// createTensorLayer returns multiple layers if quantizing (weight + scales)
			newLayers, err := createTensorLayer(td.SafetensorsReader(), tensorName, td.Dtype, td.Shape, quantizeType)
			if err != nil {
				extractor.Close()
				return fmt.Errorf("failed to create layer for %s: %w", tensorName, err)
			}
			*layers = append(*layers, newLayers...)
		}

		extractor.Close()
	}
	return nil
}

// processConfigFiles processes all .json config files in the directory
func processConfigFiles(entries []os.DirEntry, modelDir string, createLayer LayerCreator, layers *[]LayerInfo, configLayer *LayerInfo, fn func(status string)) error {
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		// Skip the index file as we don't need it after extraction
		if entry.Name() == "model.safetensors.index.json" {
			continue
		}

		cfgPath := entry.Name()
		fullPath := filepath.Join(modelDir, cfgPath)

		fn(fmt.Sprintf("importing config %s", cfgPath))

		f, err := os.Open(fullPath)
		if err != nil {
			return fmt.Errorf("failed to open %s: %w", cfgPath, err)
		}

		layer, err := createLayer(f, "application/vnd.ollama.image.json", cfgPath)
		f.Close()
		if err != nil {
			return fmt.Errorf("failed to create layer for %s: %w", cfgPath, err)
		}

		// Use config.json as the config layer
		if cfgPath == "config.json" {
			*configLayer = layer
		}

		*layers = append(*layers, layer)
	}
	return nil
}
