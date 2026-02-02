// preprocessor.go - Parser fuer HuggingFace preprocessor_config.json
//
// Extrahiert Bildvorverarbeitungs-Parameter wie image_mean, image_std.
//
// Autor: Agent 2 - Phase 9
// Datum: 2026-02-01
package huggingface

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// Resampling-Konstanten (PIL/Pillow Werte)
const (
	ResampleNearest  = 0
	ResampleLanczos  = 1
	ResampleBilinear = 2
	ResampleBicubic  = 3
)

// Fehler
var (
	ErrPreprocessorNotFound = errors.New("preprocessor_config.json nicht gefunden")
	ErrInvalidPreprocessor  = errors.New("ungueltige preprocessor_config.json")
)

// ParsePreprocessorConfig parst JSON-Bytes einer preprocessor_config.json.
func ParsePreprocessorConfig(data []byte) (*PreprocessorConfig, error) {
	var config PreprocessorConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, &HuggingFaceError{Op: "parse_preprocessor", Err: fmt.Errorf("%w: %v", ErrInvalidPreprocessor, err)}
	}
	if !hasValidPreprocessorData(&config) {
		return nil, &HuggingFaceError{Op: "parse_preprocessor", Err: ErrInvalidPreprocessor}
	}
	return &config, nil
}

// LoadPreprocessorConfig laedt und parst eine preprocessor_config.json.
func LoadPreprocessorConfig(path string) (*PreprocessorConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, &HuggingFaceError{Op: "load_preprocessor", Err: ErrPreprocessorNotFound}
		}
		return nil, &HuggingFaceError{Op: "load_preprocessor", Err: fmt.Errorf("lesen: %w", err)}
	}
	return ParsePreprocessorConfig(data)
}

// GetImageMean gibt die Normalisierungs-Mittelwerte zurueck.
func GetImageMean(config *PreprocessorConfig) []float32 {
	if config == nil {
		return DefaultImageMeanImageNet
	}
	if len(config.ImageMean) >= 3 {
		return config.ImageMean
	}
	return getDefaultMeanForProcessor(config)
}

// GetImageStd gibt die Normalisierungs-Standardabweichungen zurueck.
func GetImageStd(config *PreprocessorConfig) []float32 {
	if config == nil {
		return DefaultImageStdImageNet
	}
	if len(config.ImageStd) >= 3 {
		return config.ImageStd
	}
	return getDefaultStdForProcessor(config)
}

// GetImageSize gibt Breite und Hoehe des Eingabebildes zurueck.
func GetImageSize(config *PreprocessorConfig) (width, height int) {
	if config == nil {
		return DefaultImageSize, DefaultImageSize
	}
	if config.Size != nil {
		if w, h := extractSizeFromBlock(config.Size); w > 0 && h > 0 {
			return w, h
		}
	}
	if config.CropSize != nil {
		if w, h := extractSizeFromBlock(config.CropSize); w > 0 && h > 0 {
			return w, h
		}
	}
	if config.ImageSizeDirect > 0 {
		return config.ImageSizeDirect, config.ImageSizeDirect
	}
	if config.Width > 0 && config.Height > 0 {
		return config.Width, config.Height
	}
	return DefaultImageSize, DefaultImageSize
}

// GetRescaleFactor gibt den Rescale-Faktor zurueck (Standard: 1/255).
func GetRescaleFactor(config *PreprocessorConfig) float32 {
	if config == nil || config.RescaleFactor == 0 {
		return 1.0 / 255.0
	}
	return config.RescaleFactor
}

// GetResampleMethod gibt die Resampling-Methode als String zurueck.
func GetResampleMethod(config *PreprocessorConfig) string {
	if config == nil {
		return "bilinear"
	}
	methods := map[int]string{ResampleNearest: "nearest", ResampleLanczos: "lanczos",
		ResampleBilinear: "bilinear", ResampleBicubic: "bicubic"}
	if m, ok := methods[config.Resample]; ok {
		return m
	}
	return "bilinear"
}

// ShouldResize prueft ob Resize durchgefuehrt werden soll.
func ShouldResize(config *PreprocessorConfig) bool {
	return config == nil || config.DoResize
}

// ShouldCenterCrop prueft ob Center-Crop durchgefuehrt werden soll.
func ShouldCenterCrop(config *PreprocessorConfig) bool {
	return config != nil && config.DoCenterCrop
}

// ShouldNormalize prueft ob Normalisierung durchgefuehrt werden soll.
func ShouldNormalize(config *PreprocessorConfig) bool {
	return config == nil || config.DoNormalize
}

// hasValidPreprocessorData prueft ob mindestens ein gueltiges Feld gesetzt ist.
func hasValidPreprocessorData(config *PreprocessorConfig) bool {
	hasSize := config.Size != nil || config.CropSize != nil || config.ImageSizeDirect > 0 || config.Width > 0
	return hasSize || len(config.ImageMean) > 0 || len(config.ImageStd) > 0 ||
		config.ImageProcessorType != "" || config.ProcessorClass != ""
}

// extractSizeFromBlock extrahiert Breite und Hoehe aus einem ImageSizeConfig.
func extractSizeFromBlock(block *ImageSizeConfig) (width, height int) {
	if block == nil {
		return 0, 0
	}
	if block.Width > 0 && block.Height > 0 {
		return block.Width, block.Height
	}
	if block.ShortestEdge > 0 {
		return block.ShortestEdge, block.ShortestEdge
	}
	if block.LongestEdge > 0 {
		return block.LongestEdge, block.LongestEdge
	}
	return 0, 0
}

// getDefaultMeanForProcessor gibt Standard-Mean basierend auf Processor-Typ.
func getDefaultMeanForProcessor(config *PreprocessorConfig) []float32 {
	pt := config.ImageProcessorType + config.ProcessorClass
	if containsAny(pt, "SigLIP", "Siglip", "siglip") {
		return DefaultImageMeanSigLIP
	}
	if containsAny(pt, "CLIP", "Clip", "clip") {
		return DefaultImageMeanCLIP
	}
	return DefaultImageMeanImageNet
}

// getDefaultStdForProcessor gibt Standard-Std basierend auf Processor-Typ.
func getDefaultStdForProcessor(config *PreprocessorConfig) []float32 {
	pt := config.ImageProcessorType + config.ProcessorClass
	if containsAny(pt, "SigLIP", "Siglip", "siglip") {
		return DefaultImageStdSigLIP
	}
	if containsAny(pt, "CLIP", "Clip", "clip") {
		return DefaultImageStdCLIP
	}
	return DefaultImageStdImageNet
}

// LoadPreprocessorFromDir laedt die preprocessor_config.json aus einem Verzeichnis.
func LoadPreprocessorFromDir(dirPath string) (*PreprocessorConfig, error) {
	return LoadPreprocessorConfig(dirPath + "/preprocessor_config.json")
}

// GetNormalizationForModelType gibt Normalisierungswerte fuer einen Modell-Typ.
func GetNormalizationForModelType(modelType string) (mean, std []float32) {
	switch modelType {
	case ModelTypeSigLIP, ModelTypeSigLIP2:
		return DefaultImageMeanSigLIP, DefaultImageStdSigLIP
	case ModelTypeCLIP, ModelTypeOpenCLIP, ModelTypeEVACLIP:
		return DefaultImageMeanCLIP, DefaultImageStdCLIP
	default:
		return DefaultImageMeanImageNet, DefaultImageStdImageNet
	}
}
