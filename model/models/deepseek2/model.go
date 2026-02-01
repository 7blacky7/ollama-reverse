// Modul: model.go
// Beschreibung: DeepSeek2 Modell-Definition und Initialisierung
// Hauptstrukturen:
//   - Model: Hauptstruktur des DeepSeek2-Modells
//   - New: Erstellt ein neues DeepSeek2-Modell aus der Konfiguration
//   - Forward: Fuehrt den Vorwaertsdurchlauf des gesamten Modells durch

package deepseek2

// Verwendet DeepSeek 2 Architektur, basierend auf DeepSeek 3 Modell

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// Model repraesentiert das vollstaendige DeepSeek2-Modell
type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`

	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	*Options
}

// New erstellt ein neues DeepSeek2-Modell aus der gegebenen Konfiguration
func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count"))
	for i := range layers {
		if i < firstDenseLayerIndex {
			layers[i].MLP = &dense{}
		} else {
			layers[i].MLP = &sparse{}
		}
	}

	mScale := float32(1.0 + float64(c.Float("rope.scaling.yarn_log_multiplier"))*math.Log(float64(c.Float("rope.scaling.factor"))))
	kqScale := float64(mScale) * float64(mScale) / math.Sqrt(float64(c.Uint("attention.key_length")))

	isMLA := c.Uint("attention.key_length_mla") != 0 && c.Uint("attention.value_length_mla") != 0
	keyLength := int(cmp.Or(c.Uint("attention.key_length_mla"), c.Uint("attention.key_length")))
	valueLength := int(cmp.Or(c.Uint("attention.value_length_mla"), c.Uint("attention.value_length")))

	pre, err := buildTokenizerPatterns(c)
	if err != nil {
		return nil, err
	}

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			pre...,
		),
		Layers: layers,
		Options: &Options{
			isMLA:          isMLA,
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
			ropeScale:      c.Float("rope.scaling.factor", 1),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("expert_weights_norm", true),

			qLoraRank:     int(c.Uint("attention.q_lora_rank")),
			kvLoraRank:    int(c.Uint("attention.kv_lora_rank")),
			qkHeadDim:     keyLength,
			vHeadDim:      valueLength,
			qkRopeHeadDim: int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),

			routedScalingFactor:   c.Float("expert_weights_scale"),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),

			kqScale: kqScale,
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

// buildTokenizerPatterns erstellt die Tokenizer-Regex-Patterns basierend auf der Konfiguration
func buildTokenizerPatterns(c fs.Config) ([]string, error) {
	switch c.String("tokenizer.ggml.pre") {
	case "deepseek-v3":
		return []string{
			// Regex in mehrere Teile aufgeteilt (gemaess DeepSeek3 Regex)
			"\\p{N}{1,3}",
			`[一-龥぀-ゟ゠-ヿ]+`,
			"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
		}, nil
	case "deepseek-llm":
		// TODO: Diese Modelle wurden noch nicht geprueft, daher vorerst uebersprungen
		fallthrough
	default:
		return nil, model.ErrUnsupportedTokenizer
	}
}

// Shift verschiebt die RoPE-Positionen fuer den KV-Cache
func (m Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

// Forward fuehrt den vollstaendigen Vorwaertsdurchlauf des Modells durch
func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseek2", New)
}
