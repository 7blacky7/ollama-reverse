// detect.go - Model-Type Detection aus HuggingFace config.json
//
// Erkennt den Modell-Typ anhand der config.json Struktur.
//
// Autor: Agent 2 - Phase 9
// Datum: 2026-02-01
package huggingface

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// Fehler-Definitionen
var (
	ErrConfigNotFound   = errors.New("config.json nicht gefunden")
	ErrInvalidConfig    = errors.New("ungueltige config.json Struktur")
	ErrUnknownModelType = errors.New("unbekannter model_type")
	ErrNoVisionConfig   = errors.New("keine vision_config gefunden")
)

// DetectModelType erkennt den Modell-Typ aus einer config.json Datei.
// Gibt den normalisierten Typ-String zurueck (z.B. "siglip", "clip", "dinov2").
func DetectModelType(configPath string) (string, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", &HuggingFaceError{Op: "detect", Err: ErrConfigNotFound}
		}
		return "", &HuggingFaceError{Op: "detect", Err: fmt.Errorf("lesen: %w", err)}
	}

	info, err := ParseConfig(data)
	if err != nil {
		return "", err
	}
	return normalizeModelType(info), nil
}

// ParseConfig parst die rohen JSON-Bytes einer config.json in ConfigModelInfo.
func ParseConfig(data []byte) (*ConfigModelInfo, error) {
	var info ConfigModelInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, &HuggingFaceError{Op: "parse", Err: fmt.Errorf("%w: %v", ErrInvalidConfig, err)}
	}
	if info.ModelType == "" && len(info.Architectures) == 0 {
		return nil, &HuggingFaceError{Op: "parse", Err: ErrInvalidConfig}
	}
	return &info, nil
}

// GetVisionConfig extrahiert die Vision-Konfiguration aus ConfigModelInfo.
func GetVisionConfig(info *ConfigModelInfo) (*VisionConfig, error) {
	if info == nil {
		return nil, &HuggingFaceError{Op: "vision_config", Err: errors.New("nil")}
	}
	if info.VisionConfig != nil {
		return info.VisionConfig, nil
	}
	if isVisionOnlyModel(info) {
		return buildVisionConfigFromTopLevel(info), nil
	}
	return nil, &HuggingFaceError{Op: "vision_config", ModelID: info.ModelID, Err: ErrNoVisionConfig}
}

// normalizeModelType konvertiert model_type in einen internen Typ-String.
func normalizeModelType(info *ConfigModelInfo) string {
	modelType := strings.ToLower(info.ModelType)

	// Direkte Mappings
	typeMap := map[string]string{
		"siglip": ModelTypeSigLIP, "siglip_vision_model": ModelTypeSigLIP,
		"siglip2": ModelTypeSigLIP2, "clip": ModelTypeCLIP, "clip_vision_model": ModelTypeCLIP,
		"dinov2": ModelTypeDINOv2, "vit": ModelTypeViT, "vit_model": ModelTypeViT,
		"nomic_bert": ModelTypeNomicVit, "nomicbert": ModelTypeNomicVit,
	}
	if t, ok := typeMap[modelType]; ok {
		return t
	}

	// Aus Architectures ableiten
	for _, arch := range info.Architectures {
		archLower := strings.ToLower(arch)
		switch {
		case strings.Contains(archLower, "siglip"):
			return ModelTypeSigLIP
		case strings.Contains(archLower, "clip"):
			if containsAny(archLower, "open", "laion") {
				return ModelTypeOpenCLIP
			}
			if strings.Contains(archLower, "eva") {
				return ModelTypeEVACLIP
			}
			return ModelTypeCLIP
		case strings.Contains(archLower, "dinov2"):
			return ModelTypeDINOv2
		case strings.Contains(archLower, "vit"):
			return ModelTypeViT
		case strings.Contains(archLower, "nomic"):
			return ModelTypeNomicVit
		}
	}
	if modelType != "" {
		return modelType
	}
	return "unknown"
}

// isVisionOnlyModel prueft ob das Modell ein reines Vision-Modell ist.
func isVisionOnlyModel(info *ConfigModelInfo) bool {
	modelType := strings.ToLower(info.ModelType)
	for _, t := range []string{"dinov2", "vit", "deit", "beit", "swin"} {
		if strings.Contains(modelType, t) {
			return true
		}
	}
	for _, arch := range info.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "imageclass") || strings.Contains(archLower, "visionmodel") {
			return true
		}
	}
	return info.HiddenSize > 0 && info.NumHiddenLayers > 0
}

// buildVisionConfigFromTopLevel erstellt VisionConfig aus Top-Level-Feldern.
func buildVisionConfigFromTopLevel(info *ConfigModelInfo) *VisionConfig {
	config := &VisionConfig{
		HiddenSize: info.HiddenSize, IntermediateSize: info.IntermediateSize,
		NumHiddenLayers: info.NumHiddenLayers, NumAttentionHeads: info.NumAttentionHeads,
	}
	if config.HiddenSize == 0 {
		config.HiddenSize = DefaultHiddenSize
	}
	if config.NumAttentionHeads == 0 {
		config.NumAttentionHeads = DefaultNumHeads
	}
	if config.ImageSize == 0 {
		config.ImageSize = DefaultImageSize
	}
	if config.PatchSize == 0 {
		config.PatchSize = DefaultPatchSize
	}
	return config
}

// containsAny prueft ob str mindestens einen der Substrings enthaelt.
func containsAny(str string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(str, sub) {
			return true
		}
	}
	return false
}

// DetectFromDirectory erkennt den Modell-Typ aus einem Verzeichnis.
func DetectFromDirectory(dirPath string) (string, *ConfigModelInfo, error) {
	configPath := dirPath + "/config.json"
	modelType, err := DetectModelType(configPath)
	if err != nil {
		return "", nil, err
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return modelType, nil, nil
	}
	info, err := ParseConfig(data)
	if err != nil {
		return modelType, nil, nil
	}
	return modelType, info, nil
}

// IsVisionModel prueft ob ein ConfigModelInfo ein Vision-Modell repraesentiert.
func IsVisionModel(info *ConfigModelInfo) bool {
	if info == nil {
		return false
	}
	if info.VisionConfig != nil {
		return true
	}
	modelType := strings.ToLower(info.ModelType)
	for _, t := range []string{"siglip", "clip", "dinov2", "vit", "deit", "beit", "swin", "convnext", "eva"} {
		if strings.Contains(modelType, t) {
			return true
		}
	}
	return false
}

// GetEmbeddingDimension gibt die Embedding-Dimension des Modells zurueck.
func GetEmbeddingDimension(info *ConfigModelInfo) int {
	if info.VisionConfig != nil && info.VisionConfig.HiddenSize > 0 {
		return info.VisionConfig.HiddenSize
	}
	if info.HiddenSize > 0 {
		return info.HiddenSize
	}
	return DefaultHiddenSize
}
