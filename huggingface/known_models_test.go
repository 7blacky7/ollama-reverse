// known_models_test.go - Unit Tests fuer Known Models Registry
//
// Testet LookupKnownModel, GetConvertScript und GetDefaultOptions.
//
// Autor: Agent 4 - Phase 10
// Datum: 2026-02-01
package huggingface

import "testing"

// TestLookupKnownModel testet die Suche nach bekannten Modellen
func TestLookupKnownModel(t *testing.T) {
	tests := []struct {
		modelID, expectedType string
		expectFound           bool
	}{
		{"google/siglip-base-patch16-224", ModelTypeSigLIP, true},
		{"google/siglip-so400m-patch14-384", ModelTypeSigLIP, true},
		{"nomic-ai/nomic-embed-vision-v1", ModelTypeNomicVit, true},
		{"nomic-ai/nomic-embed-vision-v1.5", ModelTypeNomicVit, true},
		{"facebook/dinov2-small", ModelTypeDINOv2, true},
		{"facebook/dinov2-base", ModelTypeDINOv2, true},
		{"facebook/dinov2-large", ModelTypeDINOv2, true},
		{"facebook/dinov2-giant", ModelTypeDINOv2, true},
		{"laion/CLIP-ViT-B-32", ModelTypeOpenCLIP, true},
		{"BAAI/EVA02-CLIP-bigE-14", ModelTypeEVACLIP, true},
		{"BAAI/EVA-CLIP-g-14", ModelTypeEVACLIP, true},
		{"openai/clip-vit-base-patch32", ModelTypeCLIP, true},
		{"unknown/random-model", "", false},
		{"", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			model, found := LookupKnownModel(tt.modelID)
			if found != tt.expectFound {
				t.Errorf("found = %v, want %v", found, tt.expectFound)
			}
			if found && model.ModelType != tt.expectedType {
				t.Errorf("ModelType = %q, want %q", model.ModelType, tt.expectedType)
			}
		})
	}
}

// TestLookupKnownModel_Patterns testet Pattern-Matching
func TestLookupKnownModel_Patterns(t *testing.T) {
	tests := []struct {
		name, modelID string
		expectFound   bool
	}{
		{"SigLIP suffix", "google/siglip-large-patch16-256", true},
		{"Nomic suffix", "nomic-ai/nomic-embed-vision-v2", true},
		{"LAION CLIP", "laion/CLIP-ViT-L-14-laion2b", true},
		{"DINOv2 reg", "facebook/dinov2-base-reg", true},
		{"DINOv2 giant reg", "facebook/dinov2-giant-reg", true},
		{"Falsches Prefix", "custom/siglip-base", false},
		{"Kein Match", "completely/different-model", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, found := LookupKnownModel(tt.modelID)
			if found != tt.expectFound {
				t.Errorf("found = %v, want %v", found, tt.expectFound)
			}
		})
	}
}

// TestGetConvertScript testet das Ermitteln des Konvertierungs-Scripts
func TestGetConvertScript(t *testing.T) {
	tests := []struct {
		modelType, expected string
	}{
		{ModelTypeSigLIP, ScriptConvertSigLIP},
		{ModelTypeSigLIP2, ScriptConvertSigLIP},
		{ModelTypeNomicVit, ScriptConvertNomicVision},
		{ModelTypeDINOv2, ScriptConvertDINOv2},
		{ModelTypeOpenCLIP, ScriptConvertOpenCLIP},
		{ModelTypeEVACLIP, ScriptConvertEVACLIP},
		{ModelTypeCLIP, ScriptConvertCLIP},
		{ModelTypeViT, ScriptConvertViT},
		{"unknown_type", "examples/llava/convert_image_encoder_to_gguf.py"},
		{"", "examples/llava/convert_image_encoder_to_gguf.py"},
	}
	for _, tt := range tests {
		t.Run(tt.modelType, func(t *testing.T) {
			if script := GetConvertScript(tt.modelType); script != tt.expected {
				t.Errorf("Got %q, want %q", script, tt.expected)
			}
		})
	}
}

// TestGetDefaultOptions testet Standard-Optionen
func TestGetDefaultOptions(t *testing.T) {
	tests := []struct {
		modelType   string
		expectMean  []float32
		expectStd   []float32
		skipText    bool
	}{
		{ModelTypeSigLIP, DefaultImageMeanSigLIP, DefaultImageStdSigLIP, false},
		{ModelTypeSigLIP2, DefaultImageMeanSigLIP, DefaultImageStdSigLIP, false},
		{ModelTypeCLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP, true},
		{ModelTypeOpenCLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP, true},
		{ModelTypeEVACLIP, DefaultImageMeanCLIP, DefaultImageStdCLIP, true},
		{ModelTypeDINOv2, DefaultImageMeanImageNet, DefaultImageStdImageNet, false},
		{ModelTypeViT, DefaultImageMeanImageNet, DefaultImageStdImageNet, false},
		{"unknown", DefaultImageMeanImageNet, DefaultImageStdImageNet, false},
	}
	for _, tt := range tests {
		t.Run(tt.modelType, func(t *testing.T) {
			opts := GetDefaultOptions(tt.modelType)
			if !opts.VisionOnly {
				t.Error("VisionOnly sollte true sein")
			}
			if opts.SkipTextEncoder != tt.skipText {
				t.Errorf("SkipTextEncoder = %v, want %v", opts.SkipTextEncoder, tt.skipText)
			}
			if !float32SliceEqual(opts.ImageMean, tt.expectMean) {
				t.Errorf("ImageMean = %v, want %v", opts.ImageMean, tt.expectMean)
			}
			if !float32SliceEqual(opts.ImageStd, tt.expectStd) {
				t.Errorf("ImageStd = %v, want %v", opts.ImageStd, tt.expectStd)
			}
			if opts.OutputType != QuantTypeF16 {
				t.Errorf("OutputType = %q, want %q", opts.OutputType, QuantTypeF16)
			}
		})
	}
}

// TestGetAllKnownPatterns testet das Abrufen aller Patterns
func TestGetAllKnownPatterns(t *testing.T) {
	patterns := GetAllKnownPatterns()
	if len(patterns) == 0 {
		t.Error("Keine Patterns gefunden")
	}
	expected := []string{"google/siglip-*", "nomic-ai/nomic-embed-vision-*", "facebook/dinov2-small"}
	for _, exp := range expected {
		found := false
		for _, p := range patterns {
			if p == exp {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Pattern %q nicht gefunden", exp)
		}
	}
}

// TestGetModelsByType testet Filtern nach Model-Typ
func TestGetModelsByType(t *testing.T) {
	tests := []struct {
		modelType string
		minCount  int
	}{
		{ModelTypeSigLIP, 2},
		{ModelTypeDINOv2, 4},
		{ModelTypeCLIP, 1},
		{"nonexistent", 0},
	}
	for _, tt := range tests {
		t.Run(tt.modelType, func(t *testing.T) {
			models := GetModelsByType(tt.modelType)
			if len(models) < tt.minCount {
				t.Errorf("Anzahl = %d, want >= %d", len(models), tt.minCount)
			}
			for _, m := range models {
				if m.ModelType != tt.modelType {
					t.Errorf("Model.Type = %q, want %q", m.ModelType, tt.modelType)
				}
			}
		})
	}
}

// TestGetModelsByTag testet Filtern nach Tags
func TestGetModelsByTag(t *testing.T) {
	tests := []struct {
		tag      string
		minCount int
	}{
		{"vision", 5},
		{"embedding", 5},
		{"multimodal", 1},
		{"nonexistent_tag", 0},
	}
	for _, tt := range tests {
		t.Run(tt.tag, func(t *testing.T) {
			models := GetModelsByTag(tt.tag)
			if len(models) < tt.minCount {
				t.Errorf("Anzahl = %d, want >= %d", len(models), tt.minCount)
			}
		})
	}
}

// TestIsKnownModel testet die Kurzform-Funktion
func TestIsKnownModel(t *testing.T) {
	tests := []struct {
		modelID  string
		expected bool
	}{
		{"google/siglip-base-patch16-224", true},
		{"facebook/dinov2-large", true},
		{"openai/clip-vit-large-patch14", true},
		{"unknown/some-model", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			if result := IsKnownModel(tt.modelID); result != tt.expected {
				t.Errorf("Got %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestGetSupportedModelTypes testet alle Model-Typen
func TestGetSupportedModelTypes(t *testing.T) {
	types := GetSupportedModelTypes()
	expected := []string{ModelTypeSigLIP, ModelTypeSigLIP2, ModelTypeCLIP,
		ModelTypeDINOv2, ModelTypeViT, ModelTypeNomicVit, ModelTypeOpenCLIP, ModelTypeEVACLIP}
	if len(types) != len(expected) {
		t.Errorf("Anzahl = %d, want %d", len(types), len(expected))
	}
}

// TestMatchPattern testet Pattern-Matching intern
func TestMatchPattern(t *testing.T) {
	tests := []struct {
		pattern, modelID string
		expected         bool
	}{
		{"facebook/dinov2-small", "facebook/dinov2-small", true},
		{"facebook/dinov2-small", "facebook/dinov2-base", false},
		{"google/siglip-*", "google/siglip-base", true},
		{"google/siglip-*", "google/siglip-large-patch16", true},
		{"google/siglip-*", "google/clip-base", false},
		{"facebook/dinov2-*-reg", "facebook/dinov2-base-reg", true},
		{"facebook/dinov2-*-reg", "facebook/dinov2-base", false},
		{"", "", true},
	}
	for _, tt := range tests {
		name := tt.pattern + "_" + tt.modelID
		t.Run(name, func(t *testing.T) {
			if result := matchPattern(tt.pattern, tt.modelID); result != tt.expected {
				t.Errorf("Got %v, want %v", result, tt.expected)
			}
		})
	}
}

// float32SliceEqual vergleicht zwei float32 Slices
func float32SliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
