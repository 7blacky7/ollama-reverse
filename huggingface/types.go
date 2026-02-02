// types.go - Zusaetzliche Typen fuer Config-Parsing und Model-Detection
//
// Enthaelt Typen fuer:
// - config.json Parsing (ConfigModelInfo, VisionConfig)
// - preprocessor_config.json Parsing (PreprocessorConfig)
// - Bekannte Modelle Registry (KnownModel)
// - Konvertierungs-Optionen (ConvertModelOptions)
//
// HINWEIS: ModelInfo und DownloadResult sind bereits in client.go/download.go
// definiert - hier verwenden wir separate Typen fuer Config-Parsing.
//
// Autor: Agent 2 - Phase 9
// Datum: 2026-02-01
package huggingface

// =============================================================================
// KONSTANTEN - MODEL TYPES
// =============================================================================

// Unterstuetzte Model-Typen fuer Vision Embeddings
const (
	ModelTypeSigLIP   = "siglip"
	ModelTypeSigLIP2  = "siglip2"
	ModelTypeCLIP     = "clip"
	ModelTypeDINOv2   = "dinov2"
	ModelTypeViT      = "vit"
	ModelTypeNomicVit = "nomic_bert"
	ModelTypeOpenCLIP = "openclip"
	ModelTypeEVACLIP  = "evaclip"
)

// Standard-Werte fuer Bildvorverarbeitung
const (
	DefaultImageSize  = 224
	DefaultPatchSize  = 14
	DefaultNumHeads   = 12
	DefaultHiddenSize = 768
)

// Quantisierungs-Typen fuer GGUF-Konvertierung
const (
	QuantTypeF32   = "f32"
	QuantTypeF16   = "f16"
	QuantTypeQ8_0  = "q8_0"
	QuantTypeQ4_KM = "q4_k_m"
)

// =============================================================================
// CONFIG PARSING TYPEN
// =============================================================================

// ConfigModelInfo enthaelt die Metadaten aus einer HuggingFace config.json.
// Unterscheidet sich von ModelInfo (client.go) die fuer API-Responses ist.
type ConfigModelInfo struct {
	// Basis-Identifikation
	ModelType     string   `json:"model_type"`
	Architectures []string `json:"architectures,omitempty"`
	ModelID       string   `json:"-"` // Nicht in JSON, wird extern gesetzt

	// Modell-Dimensionen
	HiddenSize        int `json:"hidden_size,omitempty"`
	IntermediateSize  int `json:"intermediate_size,omitempty"`
	NumHiddenLayers   int `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads int `json:"num_attention_heads,omitempty"`

	// Vision-spezifisch (eingebettet oder separat)
	VisionConfig *VisionConfig `json:"vision_config,omitempty"`

	// Text-Config (falls multimodal)
	TextConfig *TextConfig `json:"text_config,omitempty"`

	// Tokenizer-Info
	VocabSize int `json:"vocab_size,omitempty"`

	// Zusaetzliche Felder
	TorchDtype          string `json:"torch_dtype,omitempty"`
	TransformersVersion string `json:"transformers_version,omitempty"`
}

// VisionConfig enthaelt die Vision-spezifische Konfiguration
// aus dem vision_config Block der config.json
type VisionConfig struct {
	// Architektur-Parameter
	HiddenSize        int `json:"hidden_size"`
	IntermediateSize  int `json:"intermediate_size,omitempty"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumAttentionHeads int `json:"num_attention_heads"`

	// Bild-Parameter
	ImageSize   int `json:"image_size"`
	PatchSize   int `json:"patch_size"`
	NumChannels int `json:"num_channels,omitempty"`

	// Normalisierung
	LayerNormEps float64 `json:"layer_norm_eps,omitempty"`

	// DINOv2-spezifisch
	UseRegisterTokens bool `json:"use_register_tokens,omitempty"`
	NumRegisterTokens int  `json:"num_register_tokens,omitempty"`

	// SigLIP-spezifisch
	UseHeadMasking bool `json:"use_head_masking,omitempty"`
}

// TextConfig enthaelt die Text-Encoder Konfiguration
// fuer multimodale Modelle wie CLIP
type TextConfig struct {
	HiddenSize            int `json:"hidden_size"`
	NumHiddenLayers       int `json:"num_hidden_layers"`
	NumAttentionHeads     int `json:"num_attention_heads"`
	VocabSize             int `json:"vocab_size"`
	MaxPositionEmbeddings int `json:"max_position_embeddings,omitempty"`
}

// =============================================================================
// PREPROCESSOR TYPEN
// =============================================================================

// PreprocessorConfig enthaelt die Bildvorverarbeitungs-Parameter
// aus preprocessor_config.json
type PreprocessorConfig struct {
	// Typ des Preprocessors
	ImageProcessorType string `json:"image_processor_type,omitempty"`
	ProcessorClass     string `json:"processor_class,omitempty"`

	// Bildgroesse (verschiedene Formate moeglich)
	Size     *ImageSizeConfig `json:"size,omitempty"`
	CropSize *ImageSizeConfig `json:"crop_size,omitempty"`

	// Direkte Groessenangaben (Alternative)
	ImageSizeDirect int `json:"image_size,omitempty"`
	Height          int `json:"height,omitempty"`
	Width           int `json:"width,omitempty"`

	// Normalisierung
	ImageMean []float32 `json:"image_mean,omitempty"`
	ImageStd  []float32 `json:"image_std,omitempty"`

	// Resampling
	Resample     int  `json:"resample,omitempty"`
	DoResize     bool `json:"do_resize,omitempty"`
	DoCenterCrop bool `json:"do_center_crop,omitempty"`
	DoNormalize  bool `json:"do_normalize,omitempty"`
	DoRescale    bool `json:"do_rescale,omitempty"`
	DoConvertRGB bool `json:"do_convert_rgb,omitempty"`

	// Rescale-Faktor
	RescaleFactor float32 `json:"rescale_factor,omitempty"`
}

// ImageSizeConfig repraesentiert die Bildgroesse in verschiedenen Formaten
type ImageSizeConfig struct {
	Height       int `json:"height,omitempty"`
	Width        int `json:"width,omitempty"`
	ShortestEdge int `json:"shortest_edge,omitempty"`
	LongestEdge  int `json:"longest_edge,omitempty"`
}

// =============================================================================
// BEKANNTE MODELLE
// =============================================================================

// KnownModel definiert ein bekanntes HuggingFace Vision-Modell
// mit allen relevanten Metadaten fuer die Konvertierung
type KnownModel struct {
	// Identifikation
	Pattern   string // Glob-Pattern fuer Model-ID (z.B. "google/siglip-*")
	ModelType string // Interner Typ-Identifikator

	// Konvertierung
	ConvertScript string              // Python-Script fuer GGUF-Konvertierung
	DefaultOpts   ConvertModelOptions // Standard-Optionen

	// Metadaten
	Description string   // Kurzbeschreibung
	Tags        []string // Kategorien (vision, text, multimodal)

	// Defaults fuer Preprocessing (falls nicht in config)
	DefaultImageMean []float32
	DefaultImageStd  []float32
	DefaultImageSize int
}

// =============================================================================
// KONVERTIERUNGS-OPTIONEN
// =============================================================================

// ConvertModelOptions steuert die GGUF-Konvertierung.
// Unterscheidet sich von ConvertOptions (converter.go) durch
// zusaetzliche Model-spezifische Felder.
type ConvertModelOptions struct {
	// Ausgabe
	OutputPath string // Pfad fuer GGUF-Datei
	OutputType string // Quantisierungstyp (f32, f16, q8_0, q4_k_m)

	// Vision-spezifisch
	VisionOnly bool      // Nur Vision-Encoder konvertieren
	ImageMean  []float32 // Ueberschreibt config
	ImageStd   []float32 // Ueberschreibt config
	ImageSize  int       // Ueberschreibt config

	// Modell-spezifisch
	UseRegisterTokens bool // Fuer DINOv2
	SkipTextEncoder   bool // Fuer CLIP-Varianten

	// Fortschritt
	Verbose      bool                   // Debug-Ausgaben
	ProgressFunc func(progress float64) // Fortschritts-Callback
}

// =============================================================================
// ERROR TYPEN
// =============================================================================

// HuggingFaceError repraesentiert einen Fehler bei HF-Operationen
type HuggingFaceError struct {
	Op      string // Operation (download, detect, convert)
	ModelID string // Betroffenes Modell
	Err     error  // Urspruenglicher Fehler
}

// Error implementiert das error Interface
func (e *HuggingFaceError) Error() string {
	if e.ModelID != "" {
		return "huggingface " + e.Op + " [" + e.ModelID + "]: " + e.Err.Error()
	}
	return "huggingface " + e.Op + ": " + e.Err.Error()
}

// Unwrap ermoeglicht errors.Is/As
func (e *HuggingFaceError) Unwrap() error {
	return e.Err
}
