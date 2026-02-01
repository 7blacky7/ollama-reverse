// Modul: types.go
// Beschreibung: Typen-Definitionen für das Create-Paket.
// Enthält: ModelConfig, Manifest, ManifestLayer, LayerInfo und Creator-Funktionstypen.

package create

import (
	"io"
)

// ModelConfig represents the config blob stored with a model.
type ModelConfig struct {
	ModelFormat  string   `json:"model_format"`
	Capabilities []string `json:"capabilities"`
}

// Manifest represents the manifest JSON structure.
type Manifest struct {
	SchemaVersion int             `json:"schemaVersion"`
	MediaType     string          `json:"mediaType"`
	Config        ManifestLayer   `json:"config"`
	Layers        []ManifestLayer `json:"layers"`
}

// ManifestLayer represents a layer in the manifest.
type ManifestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	Name      string `json:"name,omitempty"`
}

// LayerInfo holds metadata for a created layer.
type LayerInfo struct {
	Digest    string
	Size      int64
	MediaType string
	Name      string // Path-style name: "component/tensor" or "path/to/config.json"
}

// LayerCreator is called to create a blob layer.
// name is the path-style name (e.g., "tokenizer/tokenizer.json")
type LayerCreator func(r io.Reader, mediaType, name string) (LayerInfo, error)

// TensorLayerCreator creates a tensor blob layer with metadata.
// name is the path-style name including component (e.g., "text_encoder/model.embed_tokens.weight")
type TensorLayerCreator func(r io.Reader, name, dtype string, shape []int32) (LayerInfo, error)

// QuantizingTensorLayerCreator creates tensor layers with optional quantization.
// When quantize is non-empty (e.g., "fp8"), returns multiple layers (weight + scales + biases).
type QuantizingTensorLayerCreator func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error)

// ManifestWriter writes the manifest file.
type ManifestWriter func(modelName string, config LayerInfo, layers []LayerInfo) error
