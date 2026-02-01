// convert_types.go - Basis-Typen fuer Model-Konvertierung
// Haupttypen: ModelParameters, AdapterParameters, KV, ModelKV, ModelConverter, AdapterConverter
package convert

import (
	"fmt"
	"io/fs"
	"iter"
	"maps"
	"strings"

	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
)

// ModelParameters - Konfiguration aus config.json
type ModelParameters struct {
	Architectures []string `json:"architectures"`
	VocabSize     uint32   `json:"vocab_size"`

	// TODO is this needed?
	ModelType string `json:"model_type"`

	TextModel struct {
		VocabSize  uint32 `json:"vocab_size"`
		HiddenSize uint32 `json:"hidden_size"`
		ModelType  string `json:"model_type"`
	} `json:"text_config"`
}

// AdapterParameters - Konfiguration aus adapter_config.json
type AdapterParameters struct {
	Alpha          uint32 `json:"lora_alpha"`
	LoraLayers     uint32 `json:"lora_layers"`
	LoraParameters struct {
		Rank  uint32  `json:"rank"`
		Alpha float32 `json:"alpha"`
		Scale float32 `json:"scale"`
	} `json:"lora_parameters"`
}

// KV - Key-Value Map fuer GGUF Metadaten
type KV map[string]any

// Architecture - Gibt die Modell-Architektur zurueck
func (kv KV) Architecture() string {
	return kv.String("general.architecture", "unknown")
}

// valueTypes - Erlaubte Einzelwert-Typen fuer KV
type valueTypes interface {
	uint8 | int8 | uint16 | int16 |
		uint32 | int32 | uint64 | int64 |
		string | float32 | float64 | bool
}

// arrayValueTypes - Erlaubte Array-Typen fuer KV
type arrayValueTypes interface {
	[]uint8 | []int8 | []uint16 | []int16 |
		[]uint32 | []int32 | []uint64 | []int64 |
		[]string | []float32 | []float64 | []bool
}

// keyValue - Generische Funktion zum Abrufen von Werten aus KV
func keyValue[T valueTypes | arrayValueTypes](kv KV, key string, defaultValue ...T) (T, bool) {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key].(T); ok {
		return val, true
	}
	return defaultValue[0], false
}

// Len - Anzahl der Eintraege
func (kv KV) Len() int {
	return len(kv)
}

// Keys - Gibt alle Schluessel zurueck
func (kv KV) Keys() iter.Seq[string] {
	return maps.Keys(kv)
}

// Value - Gibt einen Wert zurueck
func (kv KV) Value(key string) any {
	return kv[key]
}

// KV - Erstellt KV-Map aus ModelParameters und Tokenizer
func (ModelParameters) KV(t *Tokenizer) KV {
	kv := KV{
		"general.file_type":            uint32(1),
		"general.quantization_version": uint32(2),
		"tokenizer.ggml.pre":           t.Pre,
		"tokenizer.ggml.model":         t.Vocabulary.Model,
		"tokenizer.ggml.tokens":        t.Vocabulary.Tokens,
		"tokenizer.ggml.scores":        t.Vocabulary.Scores,
		"tokenizer.ggml.token_type":    t.Vocabulary.Types,
	}

	if len(t.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = t.Merges
	}

	if t.Template != "" {
		kv["tokenizer.chat_template"] = t.Template
	}

	for _, sv := range t.SpecialVocabulary {
		kv[fmt.Sprintf("tokenizer.ggml.add_%s_token", sv.Key())] = sv.AddToken
		kv[fmt.Sprintf("tokenizer.ggml.%s_token_id", sv.Key())] = uint32(sv.ID)
		if len(sv.IDs) > 0 {
			kv[fmt.Sprintf("tokenizer.ggml.%s_token_ids", sv.Key())] = sv.IDs
		}
	}

	return kv
}

// KV - Erstellt KV-Map aus AdapterParameters
func (p AdapterParameters) KV() KV {
	var alpha float32
	if p.LoraParameters.Alpha == 0 {
		alpha = float32(p.Alpha)
	} else {
		alpha = p.LoraParameters.Alpha
	}

	kv := KV{
		"adapter.lora.alpha": alpha,
		"adapter.type":       "lora",
		"general.file_type":  uint32(1),
		"general.type":       "adapter",
		"general.version":    "v0.2",
	}

	return kv
}

// specialTokenTypes - Liste der speziellen Token-Typen
func (ModelParameters) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

// ModelKV - Interface fuer Model-KV-Mapping
type ModelKV interface {
	// KV maps parameters to LLM key-values
	KV(*Tokenizer) KV
}

// ModelConverter - Interface fuer Model-Konvertierung
type ModelConverter interface {
	ModelKV

	// Tensors maps input tensors to LLM tensors. Model specific modifications can be done here.
	Tensors([]Tensor) []*ggml.Tensor
	// Replacements returns a list of string pairs to replace in tensor names.
	// See [strings.Replacer](https://pkg.go.dev/strings#Replacer) for details
	Replacements() []string

	// specialTokenTypes returns any special token types the model uses
	specialTokenTypes() []string
}

// moreParser - Interface fuer zusaetzliches Parsen
type moreParser interface {
	parseMore(fs.FS) error
}

// AdapterConverter - Interface fuer Adapter-Konvertierung
type AdapterConverter interface {
	// KV maps parameters to LLM key-values
	KV(ofs.Config) KV
	// Tensors maps input tensors to LLM tensors. Adapter specific modifications can be done here.
	Tensors([]Tensor) []*ggml.Tensor
	// Replacements returns a list of string pairs to replace in tensor names.
	// See [strings.Replacer](https://pkg.go.dev/strings#Replacer) for details
	Replacements() []string
}
