// convert_model.go - Model-Konvertierung: Laedt und konvertiert Modelle zu GGUF
// Hauptfunktionen: LoadModelMetadata, ConvertModel, writeFile
package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// LoadModelMetadata - Laedt Model-Metadaten und Tokenizer aus Dateisystem
func LoadModelMetadata(fsys fs.FS) (ModelKV, *Tokenizer, error) {
	bts, err := fs.ReadFile(fsys, "config.json")
	if err != nil {
		return nil, nil, err
	}

	var p ModelParameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return nil, nil, err
	}

	if len(p.Architectures) < 1 {
		return nil, nil, errors.New("unknown architecture")
	}

	conv := createModelConverter(p.Architectures[0])
	if conv == nil {
		return nil, nil, fmt.Errorf("unsupported architecture %q", p.Architectures[0])
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return nil, nil, err
	}

	if t, ok := conv.(moreParser); ok {
		if err := t.parseMore(fsys); err != nil {
			return nil, nil, err
		}
	}

	t, err := parseTokenizer(fsys, conv.specialTokenTypes())
	if err != nil {
		return nil, nil, err
	}

	vocabSize := int(cmp.Or(p.VocabSize, p.TextModel.VocabSize))

	switch {
	case vocabSize == 0:
		slog.Debug("vocabulary size was not explicitly set by the model", "default size", len(t.Vocabulary.Tokens))
	case vocabSize > len(t.Vocabulary.Tokens):
		slog.Debug("vocabulary is smaller than expected, padding with dummy tokens", "expect", vocabSize, "actual", len(t.Vocabulary.Tokens))
		for i := range vocabSize - len(t.Vocabulary.Tokens) {
			t.Vocabulary.Tokens = append(t.Vocabulary.Tokens, fmt.Sprintf("[PAD%d]", i))
			t.Vocabulary.Scores = append(t.Vocabulary.Scores, -1)
			t.Vocabulary.Types = append(t.Vocabulary.Types, tokenTypeUserDefined)
		}
	case vocabSize < len(t.Vocabulary.Tokens):
		slog.Debug("vocabulary is larger than expected", "want", vocabSize, "got", len(t.Vocabulary.Tokens))
		p.VocabSize = uint32(len(t.Vocabulary.Tokens))
		p.TextModel.VocabSize = uint32(len(t.Vocabulary.Tokens))
	default:
		slog.Debug("vocabulary", "size", len(t.Vocabulary.Tokens))
	}
	return conv, t, nil
}

// createModelConverter - Factory fuer Model-Converter basierend auf Architektur
func createModelConverter(arch string) ModelConverter {
	switch arch {
	case "LlamaForCausalLM":
		return &llamaModel{}
	case "MllamaForConditionalGeneration":
		return &mllamaModel{}
	case "Llama4ForConditionalGeneration":
		return &llama4Model{}
	case "Mistral3ForConditionalGeneration":
		return &mistral3Model{}
	case "Ministral3ForCausalLM":
		return &mistral3CausalModel{}
	case "MixtralForCausalLM":
		return &mixtralModel{}
	case "GemmaForCausalLM":
		return &gemmaModel{}
	case "Gemma2ForCausalLM":
		return &gemma2Model{}
	case "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration":
		return &gemma3Model{Architecture: arch}
	case "Gemma3nForConditionalGeneration":
		return &gemma3nModel{}
	case "Phi3ForCausalLM":
		return &phi3Model{}
	case "Qwen2ForCausalLM":
		return &qwen2Model{}
	case "Qwen2_5_VLForConditionalGeneration":
		return &qwen25VLModel{}
	case "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration":
		return &qwen3VLModel{}
	case "Olmo3ForCausalLM":
		return &olmoModel{}
	case "BertModel":
		return &bertModel{}
	case "NomicBertModel", "NomicBertMoEModel":
		return &nomicbertModel{}
	case "CohereForCausalLM":
		return &commandrModel{}
	case "GptOssForCausalLM":
		return &gptossModel{}
	case "DeepseekOCRForCausalLM":
		return &deepseekocr{}
	case "DeepseekV3ForCausalLM":
		return &deepseek2Model{}
	case "Glm4MoeLiteForCausalLM":
		return &glm4MoeLiteModel{}
	case "Lfm2ForCausalLM":
		return &lfm2Model{}
	default:
		return nil
	}
}

// ConvertModel - Konvertiert ein Modell zu GGUF Format
// Unterstuetzte Eingabeformate: safetensors
// Unterstuetzte Tokenizer: tokenizer.json (bevorzugt), tokenizer.model
func ConvertModel(fsys fs.FS, f *os.File) error {
	kv, t, err := LoadModelMetadata(fsys)
	if err != nil {
		return err
	}
	conv := kv.(ModelConverter)

	ts, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if err != nil {
		return err
	}

	return writeFile(f, conv.KV(t), conv.Tensors(ts))
}

// writeFile - Schreibt GGUF-Datei mit KV-Metadaten und Tensoren
func writeFile(f *os.File, kv KV, ts []*ggml.Tensor) error {
	for i := range ts {
		ts[i].Shape = slices.Clone(ts[i].Shape)
		slices.Reverse(ts[i].Shape)
	}
	return ggml.WriteGGUF(f, kv, ts)
}
