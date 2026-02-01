// Package ggml - KV (Key-Value) Metadaten
//
// Dieses Modul enthaelt den KV-Typ und alle zugehoerigen Methoden:
// - KV: Map fuer GGUF Key-Value Metadaten
// - Architektur-spezifische Methoden (Architecture, FileType, BlockCount, etc.)
// - SSM-Parameter (SSMConvKernel, SSMInnerSize, etc.)
// - Generische Getter (String, Uint, Float, Bool, etc.)
package ggml

import (
	"iter"
	"log/slog"
	"maps"
	"slices"
	"strings"
)

// KV repraesentiert GGUF Key-Value Metadaten
type KV map[string]any

// Architecture gibt die Modell-Architektur zurueck
func (kv KV) Architecture() string {
	return kv.String("general.architecture", "unknown")
}

// Kind gibt den Modell-Typ zurueck
func (kv KV) Kind() string {
	return kv.String("general.type", "unknown")
}

// ParameterCount gibt die Anzahl der Parameter zurueck
func (kv KV) ParameterCount() uint64 {
	val, _ := keyValue(kv, "general.parameter_count", uint64(0))
	return val
}

// FileType gibt den GGUF FileType zurueck
func (kv KV) FileType() FileType {
	if t := kv.Uint("general.file_type"); t > 0 {
		return FileType(t)
	}
	return FileTypeUnknown
}

// BlockCount gibt die Anzahl der Bloecke/Layer zurueck
func (kv KV) BlockCount() uint64 {
	return uint64(kv.Uint("block_count"))
}

// EmbeddingLength gibt die Embedding-Dimension zurueck
func (kv KV) EmbeddingLength() uint64 {
	return uint64(kv.Uint("embedding_length"))
}

// HeadCount gibt die Anzahl der Attention-Heads pro Layer zurueck
func (kv KV) HeadCount() []uint64 {
	headCountDefault := uint32(1)
	headCount := kv.UintOrArrayValueAsArray("attention.head_count", headCountDefault)
	if len(headCount) == 1 {
		headCountDefault = headCount[0]
	}
	nLayers := int(kv.BlockCount())
	if len(headCount) > nLayers {
		slog.Warn("got more elements of attention.head_count than layers", "len(headCount)", len(headCount), "layers", nLayers)
	}
	out := make([]uint64, nLayers)
	for i := range nLayers {
		if i >= len(headCount) {
			out[i] = uint64(headCountDefault)
		} else {
			out[i] = uint64(headCount[i])
		}
	}
	return out
}

// HeadCountMax gibt das Maximum der Attention-Heads zurueck
func (kv KV) HeadCountMax() uint64 {
	return uint64(kv.UintOrMaxArrayValue("attention.head_count", 1))
}

// HeadCountMin gibt das Minimum der Attention-Heads zurueck
func (kv KV) HeadCountMin() uint64 {
	return uint64(kv.UintOrMinArrayValue("attention.head_count", 1))
}

// HeadCountKV gibt die Anzahl der KV-Heads pro Layer zurueck
func (kv KV) HeadCountKV() []uint64 {
	headCountKVDefault := uint32(1)
	headCountKV := kv.UintOrArrayValueAsArray("attention.head_count_kv", headCountKVDefault)
	if len(headCountKV) == 1 {
		headCountKVDefault = headCountKV[0]
	}
	nLayers := int(kv.BlockCount())
	if len(headCountKV) > nLayers {
		slog.Warn("got more elements of attention.head_count than layers", "len(headCountKV)", len(headCountKV), "layers", nLayers)
	}
	out := make([]uint64, nLayers)
	for i := range nLayers {
		if i >= len(headCountKV) {
			out[i] = uint64(headCountKVDefault)
		} else {
			out[i] = uint64(headCountKV[i])
		}
	}
	return out
}

// HeadCountKVMax gibt das Maximum der KV-Heads zurueck
func (kv KV) HeadCountKVMax() uint64 {
	return uint64(kv.UintOrMaxArrayValue("attention.head_count_kv", 1))
}

// HeadCountKVMin gibt das Minimum der KV-Heads zurueck
func (kv KV) HeadCountKVMin() uint64 {
	return uint64(kv.UintOrMinArrayValue("attention.head_count_kv", 1))
}

// EmbeddingHeadCountMax berechnet die maximale Head-Dimension
func (kv KV) EmbeddingHeadCountMax() uint64 {
	if heads := kv.HeadCountMin(); heads > 0 {
		return kv.EmbeddingLength() / heads
	}
	return 0
}

// EmbeddingHeadCountK gibt die Key-Dimension pro Head zurueck
func (kv KV) EmbeddingHeadCountK() uint64 {
	return uint64(kv.Uint("attention.key_length", uint32(kv.EmbeddingHeadCountMax())))
}

// EmbeddingHeadCountV gibt die Value-Dimension pro Head zurueck
func (kv KV) EmbeddingHeadCountV() uint64 {
	return uint64(kv.Uint("attention.value_length", uint32(kv.EmbeddingHeadCountMax())))
}

// ContextLength gibt die maximale Kontextlaenge zurueck
func (kv KV) ContextLength() uint64 {
	return uint64(kv.Uint("context_length"))
}

// ChatTemplate gibt das Chat-Template zurueck
func (kv KV) ChatTemplate() string {
	return kv.String("tokenizer.chat_template")
}

// SSM Architektur-Parameter

// SSMConvKernel gibt die SSM Convolution Kernel-Groesse zurueck
func (kv KV) SSMConvKernel() uint64 {
	return uint64(kv.Uint("ssm.conv_kernel"))
}

// SSMInnerSize gibt die SSM Inner-Dimension zurueck
func (kv KV) SSMInnerSize() uint64 {
	return uint64(kv.Uint("ssm.inner_size"))
}

// SSMStateSize gibt die SSM State-Groesse zurueck
func (kv KV) SSMStateSize() uint64 {
	return uint64(kv.Uint("ssm.state_size"))
}

// SSMGroupCount gibt die SSM Group-Anzahl zurueck
func (kv KV) SSMGroupCount() uint64 {
	return uint64(kv.Uint("ssm.group_count"))
}

// Generische Getter

// String gibt einen String-Wert zurueck
func (kv KV) String(key string, defaultValue ...string) string {
	val, _ := keyValue(kv, key, append(defaultValue, "")...)
	return val
}

// Uint gibt einen uint32-Wert zurueck
func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

// Float gibt einen float32-Wert zurueck
func (kv KV) Float(key string, defaultValue ...float32) float32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

// Bool gibt einen bool-Wert zurueck
func (kv KV) Bool(key string, defaultValue ...bool) bool {
	val, _ := keyValue(kv, key, append(defaultValue, false)...)
	return val
}

// UintOrMaxArrayValue gibt das Maximum eines uint32 oder uint32-Arrays zurueck
func (kv KV) UintOrMaxArrayValue(key string, defaultValue uint32) uint32 {
	_, max := kv.UintOrArrayValue(key, defaultValue)
	return max
}

// UintOrMinArrayValue gibt das Minimum eines uint32 oder uint32-Arrays zurueck
func (kv KV) UintOrMinArrayValue(key string, defaultValue uint32) uint32 {
	min, _ := kv.UintOrArrayValue(key, defaultValue)
	return min
}

// UintOrArrayValue gibt Min und Max eines uint32 oder Arrays zurueck
func (kv KV) UintOrArrayValue(key string, defaultValue uint32) (uint32, uint32) {
	arrVal := kv.UintOrArrayValueAsArray(key, defaultValue)
	return slices.Min(arrVal), slices.Max(arrVal)
}

// UintOrArrayValueAsArray gibt den Wert als Array zurueck
func (kv KV) UintOrArrayValueAsArray(key string, defaultValue uint32) []uint32 {
	if u32, ok := keyValue(kv, key, uint32(0)); ok {
		return []uint32{u32}
	} else if u32s, ok := keyValue(kv, key, &array[uint32]{}); ok {
		return u32s.values
	} else if i32s, ok := keyValue(kv, key, &array[int32]{}); ok {
		dst := make([]uint32, len(i32s.values))
		for i, v := range i32s.values {
			if v < 0 {
				slog.Warn("array values are unexpectedly negative", "key", key, "i", i, "v", v)
			}
			dst[i] = uint32(v)
		}
		return dst
	}
	return []uint32{defaultValue}
}

// Strings gibt ein String-Array zurueck
func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	val, _ := keyValue(kv, key, &array[string]{values: append(defaultValue, []string(nil))[0]})
	return val.values
}

// Ints gibt ein int32-Array zurueck
func (kv KV) Ints(key string, defaultValue ...[]int32) []int32 {
	val, _ := keyValue(kv, key, &array[int32]{values: append(defaultValue, []int32(nil))[0]})
	return val.values
}

// Uints gibt ein uint32-Array zurueck
func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	val, _ := keyValue(kv, key, &array[uint32]{values: append(defaultValue, []uint32(nil))[0]})
	return val.values
}

// Floats gibt ein float32-Array zurueck
func (kv KV) Floats(key string, defaultValue ...[]float32) []float32 {
	val, _ := keyValue(kv, key, &array[float32]{values: append(defaultValue, []float32(nil))[0]})
	return val.values
}

// Bools gibt ein bool-Array zurueck
func (kv KV) Bools(key string, defaultValue ...[]bool) []bool {
	val, _ := keyValue(kv, key, &array[bool]{values: append(defaultValue, []bool(nil))[0]})
	return val.values
}

// Len gibt die Anzahl der KV-Paare zurueck
func (kv KV) Len() int {
	return len(kv)
}

// Keys gibt einen Iterator ueber alle Keys zurueck
func (kv KV) Keys() iter.Seq[string] {
	return maps.Keys(kv)
}

// Value gibt den Wert fuer einen Key zurueck
func (kv KV) Value(key string) any {
	return kv[key]
}

// OllamaEngineRequired prueft ob die Ollama-Engine erforderlich ist
func (kv KV) OllamaEngineRequired() bool {
	return slices.Contains([]string{
		"bert",
		"deepseek2",
		"deepseekocr",
		"gemma3",
		"gemma3n",
		"gptoss", "gpt-oss",
		"llama4",
		"mistral3",
		"mllama",
		"nomic-bert",
		"olmo3",
		"qwen25vl",
		"qwen3", "qwen3moe",
		"qwen3vl", "qwen3vlmoe",
		"glm4moelite",
		"lfm2",
	}, kv.Architecture())
}

// Type Constraints fuer keyValue

type valueTypes interface {
	uint8 | int8 | uint16 | int16 |
		uint32 | int32 | uint64 | int64 |
		string | float32 | float64 | bool
}

type arrayValueTypes interface {
	*array[uint8] | *array[int8] | *array[uint16] | *array[int16] |
		*array[uint32] | *array[int32] | *array[uint64] | *array[int64] |
		*array[string] | *array[float32] | *array[float64] | *array[bool]
}

// keyValue ist eine generische Hilfsfunktion zum Lesen von KV-Werten
func keyValue[T valueTypes | arrayValueTypes](kv KV, key string, defaultValue ...T) (T, bool) {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key].(T); ok {
		return val, true
	}

	slog.Debug("key with type not found", "key", key, "default", defaultValue[0])
	return defaultValue[0], false
}
