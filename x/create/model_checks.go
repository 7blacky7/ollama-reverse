// Modul: model_checks.go
// Beschreibung: Funktionen zur Überprüfung von Modell-Typen und -Formaten.
// Enthält: IsSafetensorsModel, IsSafetensorsLLMModel, IsImageGenModel,
//          GetModelArchitecture, IsTensorModelDir, IsSafetensorsModelDir.

package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// IsSafetensorsModel checks if a model was created with the experimental
// safetensors builder by checking the model format in the config.
func IsSafetensorsModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors"
}

// IsSafetensorsLLMModel checks if a model is a safetensors LLM model
// (has completion capability, not image generation).
func IsSafetensorsLLMModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && slices.Contains(config.Capabilities, "completion")
}

// IsImageGenModel checks if a model is an image generation model
// (has image capability).
func IsImageGenModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && slices.Contains(config.Capabilities, "image")
}

// GetModelArchitecture returns the architecture from the model's config.json layer.
func GetModelArchitecture(modelName string) (string, error) {
	manifest, err := loadManifest(modelName)
	if err != nil {
		return "", err
	}

	// Find the config.json layer
	for _, layer := range manifest.Layers {
		if layer.Name == "config.json" && layer.MediaType == "application/vnd.ollama.image.json" {
			blobName := strings.Replace(layer.Digest, ":", "-", 1)
			blobPath := filepath.Join(defaultBlobDir(), blobName)

			data, err := os.ReadFile(blobPath)
			if err != nil {
				return "", err
			}

			var cfg struct {
				Architectures []string `json:"architectures"`
				ModelType     string   `json:"model_type"`
			}
			if err := json.Unmarshal(data, &cfg); err != nil {
				return "", err
			}

			// Prefer model_type, fall back to first architecture
			if cfg.ModelType != "" {
				return cfg.ModelType, nil
			}
			if len(cfg.Architectures) > 0 {
				return cfg.Architectures[0], nil
			}
		}
	}

	return "", fmt.Errorf("architecture not found in model config")
}

// IsTensorModelDir checks if the directory contains a diffusers-style tensor model
// by looking for model_index.json, which is the standard diffusers pipeline config.
func IsTensorModelDir(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model_index.json"))
	return err == nil
}

// IsSafetensorsModelDir checks if the directory contains a standard safetensors model
// by looking for config.json and at least one .safetensors file.
func IsSafetensorsModelDir(dir string) bool {
	// Must have config.json
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}

	// Must have at least one .safetensors file
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}

	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".safetensors") {
			return true
		}
	}

	return false
}
