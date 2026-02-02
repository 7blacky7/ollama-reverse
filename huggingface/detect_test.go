// detect_test.go - Unit Tests fuer Model-Type Detection
//
// Testet DetectModelType, ParseConfig und IsVisionModel Funktionen.
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

// TestDetectModelType testet die Erkennung von Model-Typen aus config.json
func TestDetectModelType(t *testing.T) {
	tempDir := t.TempDir()
	tests := []struct {
		name, config, expected string
		wantErr                bool
	}{
		{"SigLIP", `{"model_type": "siglip"}`, ModelTypeSigLIP, false},
		{"SigLIP2", `{"model_type": "siglip2"}`, ModelTypeSigLIP2, false},
		{"CLIP", `{"model_type": "clip"}`, ModelTypeCLIP, false},
		{"DINOv2", `{"model_type": "dinov2"}`, ModelTypeDINOv2, false},
		{"ViT", `{"model_type": "vit"}`, ModelTypeViT, false},
		{"NomicBERT", `{"model_type": "nomic_bert"}`, ModelTypeNomicVit, false},
		{"SigLIP via arch", `{"architectures": ["SiglipVisionModel"]}`, ModelTypeSigLIP, false},
		{"CLIP via arch", `{"architectures": ["CLIPVisionModel"]}`, ModelTypeCLIP, false},
		{"OpenCLIP via arch", `{"architectures": ["OpenCLIPModel"]}`, ModelTypeOpenCLIP, false},
		{"EVA-CLIP via arch", `{"architectures": ["EVACLIPModel"]}`, ModelTypeEVACLIP, false},
		{"DINOv2 via arch", `{"architectures": ["Dinov2Model"]}`, ModelTypeDINOv2, false},
		{"ViT via arch", `{"architectures": ["ViTModel"]}`, ModelTypeViT, false},
		{"Nomic via arch", `{"architectures": ["NomicBertModel"]}`, ModelTypeNomicVit, false},
		{"Unbekannt", `{"model_type": "custom_model"}`, "custom_model", false},
		{"Fehlt", `{"hidden_size": 768}`, "", true},
		{"siglip_vision_model", `{"model_type": "siglip_vision_model"}`, ModelTypeSigLIP, false},
		{"clip_vision_model", `{"model_type": "clip_vision_model"}`, ModelTypeCLIP, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configPath := filepath.Join(tempDir, tt.name+"_config.json")
			if err := os.WriteFile(configPath, []byte(tt.config), 0644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}
			result, err := DetectModelType(configPath)
			if tt.wantErr {
				if err == nil {
					t.Error("Erwartete Fehler")
				}
				return
			}
			if err != nil {
				t.Errorf("Unerwarteter Fehler: %v", err)
			} else if result != tt.expected {
				t.Errorf("Got %q, want %q", result, tt.expected)
			}
		})
	}
}

// TestDetectModelType_FileNotFound testet den Fehlerfall bei fehlender Datei
func TestDetectModelType_FileNotFound(t *testing.T) {
	_, err := DetectModelType("/nicht/existierender/pfad/config.json")
	if err == nil {
		t.Fatal("Erwartete Fehler fuer nicht existierende Datei")
	}
	var hfErr *HuggingFaceError
	if !errors.As(err, &hfErr) {
		t.Errorf("Erwartete HuggingFaceError, bekam %T", err)
	}
	if !errors.Is(hfErr.Err, ErrConfigNotFound) {
		t.Errorf("Erwartete ErrConfigNotFound, bekam %v", hfErr.Err)
	}
}

// TestParseConfig testet das Parsen von config.json Inhalten
func TestParseConfig(t *testing.T) {
	tests := []struct {
		name, data, checkModel, checkArch string
		expectErr                          bool
	}{
		{"Valide SigLIP", `{"model_type": "siglip"}`, "siglip", "", false},
		{"Mit architectures", `{"architectures": ["SiglipVisionModel"]}`, "", "SiglipVisionModel", false},
		{"Invalides JSON", `{"model_type": "siglip"`, "", "", true},
		{"Leere Config", `{}`, "", "", true},
		{"Nur hidden_size", `{"hidden_size": 768}`, "", "", true},
		{"Mit VisionConfig", `{"model_type": "siglip", "vision_config": {"hidden_size": 1152}}`, "siglip", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, err := ParseConfig([]byte(tt.data))
			if tt.expectErr {
				if err == nil {
					t.Error("Erwartete Fehler")
				}
				return
			}
			if err != nil {
				t.Errorf("Unerwarteter Fehler: %v", err)
				return
			}
			if tt.checkModel != "" && info.ModelType != tt.checkModel {
				t.Errorf("ModelType = %q, erwartet %q", info.ModelType, tt.checkModel)
			}
			if tt.checkArch != "" && (len(info.Architectures) == 0 || info.Architectures[0] != tt.checkArch) {
				t.Errorf("Architecture falsch, erwartet %q", tt.checkArch)
			}
		})
	}
}

// TestIsVisionModel testet die Vision-Model Erkennung
func TestIsVisionModel(t *testing.T) {
	tests := []struct {
		name     string
		info     *ConfigModelInfo
		expected bool
	}{
		{"Nil", nil, false},
		{"SigLIP", &ConfigModelInfo{ModelType: "siglip"}, true},
		{"CLIP", &ConfigModelInfo{ModelType: "clip"}, true},
		{"DINOv2", &ConfigModelInfo{ModelType: "dinov2"}, true},
		{"ViT", &ConfigModelInfo{ModelType: "vit"}, true},
		{"EVA", &ConfigModelInfo{ModelType: "eva"}, true},
		{"Swin", &ConfigModelInfo{ModelType: "swin"}, true},
		{"ConvNext", &ConfigModelInfo{ModelType: "convnext"}, true},
		{"Mit VisionConfig", &ConfigModelInfo{VisionConfig: &VisionConfig{HiddenSize: 768}}, true},
		{"BERT", &ConfigModelInfo{ModelType: "bert"}, false},
		{"GPT", &ConfigModelInfo{ModelType: "gpt2"}, false},
		{"Llama", &ConfigModelInfo{ModelType: "llama"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := IsVisionModel(tt.info); result != tt.expected {
				t.Errorf("IsVisionModel() = %v, erwartet %v", result, tt.expected)
			}
		})
	}
}

// TestGetVisionConfig testet das Extrahieren der Vision-Konfiguration
func TestGetVisionConfig(t *testing.T) {
	tests := []struct {
		name    string
		info    *ConfigModelInfo
		wantErr bool
		checkHS int
	}{
		{"Nil", nil, true, 0},
		{"Direkt vorhanden", &ConfigModelInfo{VisionConfig: &VisionConfig{HiddenSize: 1152}}, false, 1152},
		{"Vision-only", &ConfigModelInfo{ModelType: "dinov2", HiddenSize: 768}, false, 0},
		{"Non-Vision", &ConfigModelInfo{ModelType: "bert"}, true, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := GetVisionConfig(tt.info)
			if tt.wantErr {
				if err == nil {
					t.Error("Erwartete Fehler")
				}
				return
			}
			if err != nil {
				t.Errorf("Unerwarteter Fehler: %v", err)
			} else if tt.checkHS > 0 && config.HiddenSize != tt.checkHS {
				t.Errorf("HiddenSize = %d, erwartet %d", config.HiddenSize, tt.checkHS)
			}
		})
	}
}

// TestGetEmbeddingDimension testet das Ermitteln der Embedding-Dimension
func TestGetEmbeddingDimension(t *testing.T) {
	tests := []struct {
		name     string
		info     *ConfigModelInfo
		expected int
	}{
		{"Aus VisionConfig", &ConfigModelInfo{VisionConfig: &VisionConfig{HiddenSize: 1152}}, 1152},
		{"Top-Level", &ConfigModelInfo{HiddenSize: 768}, 768},
		{"Default", &ConfigModelInfo{}, DefaultHiddenSize},
		{"VisionConfig Vorrang", &ConfigModelInfo{HiddenSize: 512, VisionConfig: &VisionConfig{HiddenSize: 1024}}, 1024},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := GetEmbeddingDimension(tt.info); result != tt.expected {
				t.Errorf("Got %d, want %d", result, tt.expected)
			}
		})
	}
}

// TestDetectFromDirectory testet die Erkennung aus einem Verzeichnis
func TestDetectFromDirectory(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.json")
	if err := os.WriteFile(configPath, []byte(`{"model_type": "siglip", "hidden_size": 1152}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	modelType, info, err := DetectFromDirectory(tempDir)
	if err != nil {
		t.Fatalf("DetectFromDirectory: %v", err)
	}
	if modelType != ModelTypeSigLIP {
		t.Errorf("ModelType = %q, erwartet %q", modelType, ModelTypeSigLIP)
	}
	if info == nil || info.HiddenSize != 1152 {
		t.Errorf("HiddenSize falsch")
	}
}
