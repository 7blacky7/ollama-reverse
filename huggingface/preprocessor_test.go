// preprocessor_test.go - Unit Tests fuer Preprocessor Config Parser
//
// Testet ParsePreprocessorConfig, GetImageMean, GetImageStd und GetImageSize.
//
// Autor: Agent 4 - Phase 10
// Datum: 2026-02-01
package huggingface

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestParsePreprocessorConfig testet das Parsen von preprocessor_config.json
func TestParsePreprocessorConfig(t *testing.T) {
	tests := []struct {
		name, data string
		expectErr  bool
	}{
		{"SigLIP", `{"image_processor_type": "SiglipImageProcessor", "size": {"height": 384, "width": 384}}`, false},
		{"CLIP crop", `{"processor_class": "CLIPProcessor", "crop_size": {"height": 224, "width": 224}}`, false},
		{"image_size direkt", `{"image_processor_type": "ViTImageProcessor", "image_size": 224}`, false},
		{"shortest_edge", `{"image_processor_type": "DINOv2Processor", "size": {"shortest_edge": 518}}`, false},
		{"Invalides JSON", `{"invalid: json}`, true},
		{"Leere Config", `{}`, true},
		{"Nur Typ", `{"image_processor_type": "Processor"}`, false},
		{"rescale_factor", `{"processor_class": "Processor", "size": {"height": 224, "width": 224}, "rescale_factor": 0.00392}`, false},
		{"resampling", `{"image_processor_type": "Processor", "size": {"height": 224, "width": 224}, "resample": 3}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := ParsePreprocessorConfig([]byte(tt.data))
			if tt.expectErr {
				if err == nil {
					t.Error("Erwartete Fehler")
				}
				return
			}
			if err != nil {
				t.Errorf("Unerwarteter Fehler: %v", err)
			} else if config == nil {
				t.Error("Erwartete Config, bekam nil")
			}
		})
	}
}

// TestLoadPreprocessorConfig testet das Laden aus einer Datei
func TestLoadPreprocessorConfig(t *testing.T) {
	tempDir := t.TempDir()
	validPath := filepath.Join(tempDir, "preprocessor_config.json")
	os.WriteFile(validPath, []byte(`{"image_processor_type": "Test", "size": {"height": 384, "width": 384}}`), 0644)

	t.Run("Valide Datei", func(t *testing.T) {
		config, err := LoadPreprocessorConfig(validPath)
		if err != nil {
			t.Fatalf("Fehler: %v", err)
		}
		if config.ImageProcessorType != "Test" {
			t.Errorf("ImageProcessorType = %q, erwartet Test", config.ImageProcessorType)
		}
	})

	t.Run("Nicht existierend", func(t *testing.T) {
		_, err := LoadPreprocessorConfig("/nicht/vorhanden.json")
		var hfErr *HuggingFaceError
		if !errors.As(err, &hfErr) {
			t.Errorf("Erwartete HuggingFaceError, bekam %T", err)
		}
	})
}

// TestGetImageMean testet Normalisierungs-Mittelwerte
func TestGetImageMean(t *testing.T) {
	tests := []struct {
		name     string
		config   *PreprocessorConfig
		expected []float32
	}{
		{"Nil", nil, DefaultImageMeanImageNet},
		{"Explizit", &PreprocessorConfig{ImageMean: []float32{0.5, 0.5, 0.5}}, []float32{0.5, 0.5, 0.5}},
		{"SigLIP Processor", &PreprocessorConfig{ImageProcessorType: "SiglipImageProcessor"}, DefaultImageMeanSigLIP},
		{"CLIP Processor", &PreprocessorConfig{ProcessorClass: "CLIPImageProcessor"}, DefaultImageMeanCLIP},
		{"Zu kurz", &PreprocessorConfig{ImageMean: []float32{0.5, 0.5}}, DefaultImageMeanImageNet},
		{"Leer", &PreprocessorConfig{ImageMean: []float32{}}, DefaultImageMeanImageNet},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetImageMean(tt.config)
			if !float32SliceEqual(result, tt.expected) {
				t.Errorf("Got %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestGetImageStd testet Standardabweichungen
func TestGetImageStd(t *testing.T) {
	tests := []struct {
		name     string
		config   *PreprocessorConfig
		expected []float32
	}{
		{"Nil", nil, DefaultImageStdImageNet},
		{"Explizit", &PreprocessorConfig{ImageStd: []float32{0.5, 0.5, 0.5}}, []float32{0.5, 0.5, 0.5}},
		{"SigLIP Processor", &PreprocessorConfig{ImageProcessorType: "SiglipImageProcessor"}, DefaultImageStdSigLIP},
		{"CLIP Processor", &PreprocessorConfig{ProcessorClass: "CLIPProcessor"}, DefaultImageStdCLIP},
		{"Zu kurz", &PreprocessorConfig{ImageStd: []float32{0.5}}, DefaultImageStdImageNet},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetImageStd(tt.config)
			if !float32SliceEqual(result, tt.expected) {
				t.Errorf("Got %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestGetImageSize testet das Ermitteln der Bildgroesse
func TestGetImageSize(t *testing.T) {
	tests := []struct {
		name           string
		config         *PreprocessorConfig
		expectW, expectH int
	}{
		{"Nil", nil, DefaultImageSize, DefaultImageSize},
		{"Size w/h", &PreprocessorConfig{Size: &ImageSizeConfig{Width: 384, Height: 384}}, 384, 384},
		{"shortest_edge", &PreprocessorConfig{Size: &ImageSizeConfig{ShortestEdge: 518}}, 518, 518},
		{"longest_edge", &PreprocessorConfig{Size: &ImageSizeConfig{LongestEdge: 1024}}, 1024, 1024},
		{"CropSize", &PreprocessorConfig{CropSize: &ImageSizeConfig{Width: 224, Height: 224}}, 224, 224},
		{"image_size direkt", &PreprocessorConfig{ImageSizeDirect: 336}, 336, 336},
		{"width/height direkt", &PreprocessorConfig{Width: 256, Height: 256}, 256, 256},
		{"Size Vorrang", &PreprocessorConfig{Size: &ImageSizeConfig{Width: 384, Height: 384}, CropSize: &ImageSizeConfig{Width: 224, Height: 224}}, 384, 384},
		{"Leere Size", &PreprocessorConfig{Size: &ImageSizeConfig{}, CropSize: &ImageSizeConfig{Width: 224, Height: 224}}, 224, 224},
		{"Nicht-quadratisch", &PreprocessorConfig{Size: &ImageSizeConfig{Width: 640, Height: 480}}, 640, 480},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, h := GetImageSize(tt.config)
			if w != tt.expectW || h != tt.expectH {
				t.Errorf("Got (%d, %d), want (%d, %d)", w, h, tt.expectW, tt.expectH)
			}
		})
	}
}

// TestGetRescaleFactor testet Rescale-Faktor
func TestGetRescaleFactor(t *testing.T) {
	tests := []struct {
		name     string
		config   *PreprocessorConfig
		expected float32
	}{
		{"Nil", nil, 1.0 / 255.0},
		{"Explizit", &PreprocessorConfig{RescaleFactor: 0.00392156862}, 0.00392156862},
		{"Null", &PreprocessorConfig{RescaleFactor: 0}, 1.0 / 255.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetRescaleFactor(tt.config)
			diff := result - tt.expected
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.0001 {
				t.Errorf("Got %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestGetResampleMethod testet Resampling-Methode
func TestGetResampleMethod(t *testing.T) {
	tests := []struct {
		name     string
		config   *PreprocessorConfig
		expected string
	}{
		{"Nil", nil, "bilinear"},
		{"Nearest", &PreprocessorConfig{Resample: ResampleNearest}, "nearest"},
		{"Lanczos", &PreprocessorConfig{Resample: ResampleLanczos}, "lanczos"},
		{"Bilinear", &PreprocessorConfig{Resample: ResampleBilinear}, "bilinear"},
		{"Bicubic", &PreprocessorConfig{Resample: ResampleBicubic}, "bicubic"},
		{"Unbekannt", &PreprocessorConfig{Resample: 99}, "bilinear"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := GetResampleMethod(tt.config); result != tt.expected {
				t.Errorf("Got %q, want %q", result, tt.expected)
			}
		})
	}
}

// TestShouldResize testet Resize-Pruefung
func TestShouldResize(t *testing.T) {
	if !ShouldResize(nil) {
		t.Error("Nil sollte true sein")
	}
	if !ShouldResize(&PreprocessorConfig{DoResize: true}) {
		t.Error("DoResize=true sollte true sein")
	}
	if ShouldResize(&PreprocessorConfig{DoResize: false}) {
		t.Error("DoResize=false sollte false sein")
	}
}

// TestShouldCenterCrop testet CenterCrop-Pruefung
func TestShouldCenterCrop(t *testing.T) {
	if ShouldCenterCrop(nil) {
		t.Error("Nil sollte false sein")
	}
	if !ShouldCenterCrop(&PreprocessorConfig{DoCenterCrop: true}) {
		t.Error("DoCenterCrop=true sollte true sein")
	}
}

// TestShouldNormalize testet Normalize-Pruefung
func TestShouldNormalize(t *testing.T) {
	if !ShouldNormalize(nil) {
		t.Error("Nil sollte true sein")
	}
	if ShouldNormalize(&PreprocessorConfig{DoNormalize: false}) {
		t.Error("DoNormalize=false sollte false sein")
	}
}

// TestGetNormalizationForModelType testet model-spezifische Normalisierung
func TestGetNormalizationForModelType(t *testing.T) {
	tests := []struct {
		modelType   string
		expectMean  []float32
		expectStd   []float32
	}{
		{ModelTypeSigLIP, DefaultImageMeanSigLIP, DefaultImageStdSigLIP},
		{ModelTypeSigLIP2, DefaultImageMeanSigLIP, DefaultImageStdSigLIP},
		{ModelTypeCLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP},
		{ModelTypeOpenCLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP},
		{ModelTypeEVACLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP},
		{ModelTypeDINOv2, DefaultImageMeanImageNet, DefaultImageStdImageNet},
		{ModelTypeViT, DefaultImageMeanImageNet, DefaultImageStdImageNet},
		{"unknown", DefaultImageMeanImageNet, DefaultImageStdImageNet},
	}
	for _, tt := range tests {
		t.Run(tt.modelType, func(t *testing.T) {
			mean, std := GetNormalizationForModelType(tt.modelType)
			if !float32SliceEqual(mean, tt.expectMean) {
				t.Errorf("Mean = %v, want %v", mean, tt.expectMean)
			}
			if !float32SliceEqual(std, tt.expectStd) {
				t.Errorf("Std = %v, want %v", std, tt.expectStd)
			}
		})
	}
}

// TestLoadPreprocessorFromDir testet Laden aus Verzeichnis
func TestLoadPreprocessorFromDir(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "preprocessor_config.json")
	os.WriteFile(configPath, []byte(`{"image_processor_type": "Test", "size": {"height": 256, "width": 256}}`), 0644)

	config, err := LoadPreprocessorFromDir(tempDir)
	if err != nil {
		t.Fatalf("Fehler: %v", err)
	}
	if config.ImageProcessorType != "Test" {
		t.Errorf("ImageProcessorType = %q, erwartet Test", config.ImageProcessorType)
	}
	w, h := GetImageSize(config)
	if w != 256 || h != 256 {
		t.Errorf("Groesse = (%d, %d), erwartet (256, 256)", w, h)
	}
}
