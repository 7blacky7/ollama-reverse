// known_models.go - Registry bekannter HuggingFace Vision-Modelle
//
// Definiert bekannte Vision Embedding Modelle mit Standard-Konfigurationen.
//
// Autor: Agent 2 - Phase 9
// Datum: 2026-02-01
package huggingface

import (
	"path/filepath"
	"strings"
)

// Pfade zu den Konvertierungs-Scripts (relativ zum Projekt-Root)
const (
	ScriptConvertSigLIP      = "scripts/convert_siglip.py"
	ScriptConvertNomicVision = "scripts/convert_nomic_vision.py"
	ScriptConvertOpenCLIP    = "scripts/convert_openclip.py"
	ScriptConvertDINOv2      = "scripts/convert_dinov2.py"
	ScriptConvertEVACLIP     = "scripts/convert_eva_clip.py"
	ScriptConvertCLIP        = "scripts/convert_clip.py"
	ScriptConvertViT         = "scripts/convert_vit.py"
)

// Standard-Normalisierungswerte
var (
	DefaultImageMeanImageNet = []float32{0.485, 0.456, 0.406}
	DefaultImageStdImageNet  = []float32{0.229, 0.224, 0.225}
	DefaultImageMeanSigLIP   = []float32{0.5, 0.5, 0.5}
	DefaultImageStdSigLIP    = []float32{0.5, 0.5, 0.5}
	DefaultImageMeanCLIP     = []float32{0.48145466, 0.4578275, 0.40821073}
	DefaultImageStdCLIP      = []float32{0.26862954, 0.26130258, 0.27577711}
)

// KnownModels enthaelt alle bekannten HuggingFace Vision-Modelle
var KnownModels = map[string]KnownModel{
	// Google SigLIP
	"google/siglip-*": newSigLIPModel("google/siglip-*", 224,
		"Google SigLIP - Sigmoid Loss Image-Text Pretraining"),
	"google/siglip-so400m-*": newSigLIPModel("google/siglip-so400m-*", 384,
		"Google SigLIP SO400M - 400M Parameter Variante"),

	// Nomic Vision
	"nomic-ai/nomic-embed-vision-*": newNomicModel("nomic-ai/nomic-embed-vision-*",
		"Nomic Embed Vision - Unified Text+Image Embedding Space"),

	// Meta DINOv2
	"facebook/dinov2-small": newDINOv2Model("facebook/dinov2-small", false,
		"Meta DINOv2 Small - Self-Supervised Vision Transformer"),
	"facebook/dinov2-base": newDINOv2Model("facebook/dinov2-base", false,
		"Meta DINOv2 Base - Self-Supervised Vision Transformer"),
	"facebook/dinov2-large": newDINOv2Model("facebook/dinov2-large", false,
		"Meta DINOv2 Large - Self-Supervised Vision Transformer"),
	"facebook/dinov2-giant": newDINOv2Model("facebook/dinov2-giant", false,
		"Meta DINOv2 Giant - 1B Parameter Vision Transformer"),
	"facebook/dinov2-*-reg": newDINOv2Model("facebook/dinov2-*-reg", true,
		"Meta DINOv2 mit Register-Tokens"),

	// LAION OpenCLIP
	"laion/CLIP-*": newCLIPModel("laion/CLIP-*", ModelTypeOpenCLIP, ScriptConvertOpenCLIP, 224,
		"LAION OpenCLIP - Open Source CLIP auf LAION-2B"),

	// BAAI EVA-CLIP
	"BAAI/EVA02-CLIP-*": newCLIPModel("BAAI/EVA02-CLIP-*", ModelTypeEVACLIP, ScriptConvertEVACLIP, 336,
		"BAAI EVA-CLIP - Enhanced Vision Architecture"),
	"BAAI/EVA-CLIP-*": newCLIPModel("BAAI/EVA-CLIP-*", ModelTypeEVACLIP, ScriptConvertEVACLIP, 224,
		"BAAI EVA-CLIP v1"),

	// OpenAI CLIP
	"openai/clip-*": newCLIPModel("openai/clip-*", ModelTypeCLIP, ScriptConvertCLIP, 224,
		"OpenAI CLIP - Contrastive Language-Image Pretraining"),
}

// Factory-Funktionen fuer bekannte Modelle
func newSigLIPModel(pattern string, imageSize int, desc string) KnownModel {
	return KnownModel{
		Pattern: pattern, ModelType: ModelTypeSigLIP, ConvertScript: ScriptConvertSigLIP,
		DefaultOpts:      ConvertModelOptions{VisionOnly: true, OutputType: QuantTypeF16},
		Description:      desc, Tags: []string{"vision", "embedding", "siglip"},
		DefaultImageMean: DefaultImageMeanSigLIP, DefaultImageStd: DefaultImageStdSigLIP,
		DefaultImageSize: imageSize,
	}
}

func newNomicModel(pattern, desc string) KnownModel {
	return KnownModel{
		Pattern: pattern, ModelType: ModelTypeNomicVit, ConvertScript: ScriptConvertNomicVision,
		DefaultOpts:      ConvertModelOptions{VisionOnly: true, OutputType: QuantTypeF16},
		Description:      desc, Tags: []string{"vision", "embedding", "nomic", "multimodal"},
		DefaultImageMean: DefaultImageMeanImageNet, DefaultImageStd: DefaultImageStdImageNet,
		DefaultImageSize: 224,
	}
}

func newDINOv2Model(pattern string, useRegTokens bool, desc string) KnownModel {
	return KnownModel{
		Pattern: pattern, ModelType: ModelTypeDINOv2, ConvertScript: ScriptConvertDINOv2,
		DefaultOpts: ConvertModelOptions{
			VisionOnly: true, UseRegisterTokens: useRegTokens, OutputType: QuantTypeF16,
		},
		Description:      desc, Tags: []string{"vision", "embedding", "dinov2"},
		DefaultImageMean: DefaultImageMeanImageNet, DefaultImageStd: DefaultImageStdImageNet,
		DefaultImageSize: 224,
	}
}

func newCLIPModel(pattern, modelType, script string, imageSize int, desc string) KnownModel {
	return KnownModel{
		Pattern: pattern, ModelType: modelType, ConvertScript: script,
		DefaultOpts: ConvertModelOptions{
			VisionOnly: true, SkipTextEncoder: true, OutputType: QuantTypeF16,
		},
		Description:      desc, Tags: []string{"vision", "embedding", "clip"},
		DefaultImageMean: DefaultImageMeanCLIP, DefaultImageStd: DefaultImageStdCLIP,
		DefaultImageSize: imageSize,
	}
}

// LookupKnownModel sucht ein bekanntes Modell anhand der Model-ID
func LookupKnownModel(modelID string) (*KnownModel, bool) {
	if model, ok := KnownModels[modelID]; ok {
		return &model, true
	}
	for pattern, model := range KnownModels {
		if matchPattern(pattern, modelID) {
			return &model, true
		}
	}
	return nil, false
}

// GetConvertScript gibt das Konvertierungs-Script fuer einen Modell-Typ zurueck
func GetConvertScript(modelType string) string {
	scripts := map[string]string{
		ModelTypeSigLIP: ScriptConvertSigLIP, ModelTypeSigLIP2: ScriptConvertSigLIP,
		ModelTypeNomicVit: ScriptConvertNomicVision, ModelTypeDINOv2: ScriptConvertDINOv2,
		ModelTypeOpenCLIP: ScriptConvertOpenCLIP, ModelTypeEVACLIP: ScriptConvertEVACLIP,
		ModelTypeCLIP: ScriptConvertCLIP, ModelTypeViT: ScriptConvertViT,
	}
	if script, ok := scripts[modelType]; ok {
		return script
	}
	return "examples/llava/convert_image_encoder_to_gguf.py"
}

// GetDefaultOptions gibt Standard-Konvertierungs-Optionen zurueck
func GetDefaultOptions(modelType string) ConvertModelOptions {
	opts := ConvertModelOptions{VisionOnly: true, OutputType: QuantTypeF16}
	switch modelType {
	case ModelTypeSigLIP, ModelTypeSigLIP2:
		opts.ImageMean, opts.ImageStd = DefaultImageMeanSigLIP, DefaultImageStdSigLIP
	case ModelTypeCLIP, ModelTypeOpenCLIP, ModelTypeEVACLIP:
		opts.ImageMean, opts.ImageStd = DefaultImageMeanCLIP, DefaultImageStdCLIP
		opts.SkipTextEncoder = true
	default:
		opts.ImageMean, opts.ImageStd = DefaultImageMeanImageNet, DefaultImageStdImageNet
	}
	return opts
}

// matchPattern prueft ob eine Model-ID einem Glob-Pattern entspricht
func matchPattern(pattern, modelID string) bool {
	if !strings.Contains(pattern, "*") {
		return pattern == modelID
	}
	if strings.HasSuffix(pattern, "*") {
		return strings.HasPrefix(modelID, strings.TrimSuffix(pattern, "*"))
	}
	parts := strings.Split(pattern, "*")
	if len(parts) == 2 {
		return strings.HasPrefix(modelID, parts[0]) && strings.HasSuffix(modelID, parts[1])
	}
	matched, _ := filepath.Match(pattern, modelID)
	return matched
}

// GetAllKnownPatterns gibt alle registrierten Model-Patterns zurueck
func GetAllKnownPatterns() []string {
	patterns := make([]string, 0, len(KnownModels))
	for p := range KnownModels {
		patterns = append(patterns, p)
	}
	return patterns
}

// GetModelsByType gibt alle bekannten Modelle eines bestimmten Typs zurueck
func GetModelsByType(modelType string) []KnownModel {
	var models []KnownModel
	for _, m := range KnownModels {
		if m.ModelType == modelType {
			models = append(models, m)
		}
	}
	return models
}

// GetModelsByTag gibt alle bekannten Modelle mit einem bestimmten Tag zurueck
func GetModelsByTag(tag string) []KnownModel {
	var models []KnownModel
	for _, m := range KnownModels {
		for _, t := range m.Tags {
			if t == tag {
				models = append(models, m)
				break
			}
		}
	}
	return models
}

// IsKnownModel prueft ob eine Model-ID bekannt ist
func IsKnownModel(modelID string) bool {
	_, found := LookupKnownModel(modelID)
	return found
}

// GetSupportedModelTypes gibt alle unterstuetzten Modell-Typen zurueck
func GetSupportedModelTypes() []string {
	return []string{
		ModelTypeSigLIP, ModelTypeSigLIP2, ModelTypeCLIP, ModelTypeDINOv2,
		ModelTypeViT, ModelTypeNomicVit, ModelTypeOpenCLIP, ModelTypeEVACLIP,
	}
}
