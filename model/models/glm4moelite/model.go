// Package glm4moelite - GLM-4 MoE Lite Model Implementierung
//
// Diese Datei enthaelt:
// - Options: Konfigurationsparameter fuer das Model
// - Attention: Multi-Head Latent Attention (MLA) mit LoRA Kompression
// - Layer: Transformer-Block mit Attention und MLP
// - Model: Hauptmodel mit Embedding, Layers und Output
//
// GLM-4 MoE Lite ist ein Mixture-of-Experts Sprachmodel mit
// Multi-Head Latent Attention fuer effiziente KV-Cache Nutzung.
package glm4moelite

import (
	"errors"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// ErrOldModelFormat wird zurueckgegeben wenn ein veraltetes Weight-Format erkannt wird
var ErrOldModelFormat = errors.New("this model uses a weight format that is no longer supported; please re-download it")

// Options enthaelt alle Konfigurationsparameter fuer das GLM-4 Model
type Options struct {
	// MoE Parameter
	numExpertsUsed      int     // Anzahl aktiver Experten pro Token
	numExperts          int     // Gesamtzahl der Experten
	normTopKProb        bool    // Normalisierung der Top-K Wahrscheinlichkeiten
	routedScalingFactor float32 // Skalierungsfaktor fuer geroutete Experten

	// LoRA und Attention Dimensionen
	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	kqNopeHeadDim,
	qkHeadDim int
	qLoraRank int
	vHeadDim  int

	// Model Dimensionen
	hiddenSize,
	numHeads,
	numKVHeads int

	// Normalisierung und Positional Encoding
	eps,
	ropeBase float32
	kqScale float64
}

// applyRotaryPositionEmbeddings wendet RoPE auf den Tensor an
func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.qkRopeHeadDim, o.ropeBase, 1.0)
}

// Attention implementiert Multi-Head Latent Attention (MLA).
// Verwendet LoRA-Kompression fuer effiziente KV-Cache Nutzung.
type Attention struct {
	Q *nn.Linear `gguf:"attn_q"` // Query-Projektion (ohne LoRA)

	// Query LoRA Dekomposition
	QA     *nn.Linear  `gguf:"attn_q_a"`      // Query LoRA Down-Projektion
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"` // Query LoRA Normalisierung
	QB     *nn.Linear  `gguf:"attn_q_b"`      // Query LoRA Up-Projektion

	// Key-Value komprimierte Projektion
	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"` // KV LoRA gemeinsame Projektion
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"` // KV LoRA Normalisierung

	KB *nn.Linear `gguf:"attn_k_b"` // Key Up-Projektion
	VB *nn.Linear `gguf:"attn_v_b"` // Value Up-Projektion

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"` // Output-Projektion
}

// Forward berechnet die Multi-Head Latent Attention.
// Implementiert MLA mit Absorption: K-Projektion wird in Query absorbiert.
func (attn *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLength := hiddenStates.Dim(1)

	// Query berechnen (mit oder ohne LoRA)
	var query ml.Tensor
	if opts.qLoraRank == 0 {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	// Query in Heads aufteilen und Chunks fuer RoPE/NoRoPE erstellen
	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)
	queryChunks := query.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.qkRopeHeadDim)

	// Komprimierte KV berechnen
	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kPass := compressedKV.Slice(ctx, 0, 0, opts.kvLoraRank, 1)
	kRot := compressedKV.View(ctx,
		opts.kvLoraRank*compressedKV.Stride(0), opts.qkRopeHeadDim,
		compressedKV.Stride(1), 1,
		compressedKV.Stride(1), compressedKV.Dim(1),
	)

	// RoPE auf rotierende Teile anwenden
	qRot := opts.applyRotaryPositionEmbeddings(ctx, queryChunks[1], positions)
	kRot = opts.applyRotaryPositionEmbeddings(ctx, kRot, positions)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)

	// MLA absorption: K-Projektion in Query absorbieren
	qPass := queryChunks[0].Permute(ctx, 0, 2, 1, 3)
	qPassAbsorb := attn.KB.Forward(ctx, qPass).Permute(ctx, 0, 2, 1, 3)
	query = qRot.Concat(ctx, qPassAbsorb, 0)

	// Key zusammensetzen
	kPass = kPass.Reshape(ctx, opts.kvLoraRank, 1, seqLength)
	key := kRot.Concat(ctx, kPass, 0)

	// Attention mit Value-MLA berechnen
	attention := nn.AttentionWithVMLA(ctx, query, key, kPass, nil, attn.VB.Weight, opts.kqScale, cache)

	// Output-Projektion
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention)
}

// Layer repraesentiert einen einzelnen Transformer-Block.
// Besteht aus Attention und MLP mit Pre-Normalisierung.
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"` // Pre-Attention Normalisierung
	Attention     *Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"` // Pre-MLP Normalisierung
	MLP     MLP                           // MoE oder Dense
}

// Forward fuehrt einen Layer-Forward-Pass durch.
// Verwendet Pre-Normalisierung und Residual-Verbindungen.
func (t *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	// Optionale Output-Filterung (nur im letzten Layer)
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	// Attention Residual
	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	// MLP mit Residual
	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	return hiddenStates
}

// Model repraesentiert das vollstaendige GLM-4 MoE Lite Model
type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`  // Token Embedding
	Layers         []Layer       `gguf:"blk"`         // Transformer Layers

	OutputNorm *nn.RMSNorm `gguf:"output_norm"`           // Finale Normalisierung
	Output     *nn.Linear  `gguf:"output,alt:token_embd"` // Output Projektion (Tied Embeddings)

	*Options
}

// New erstellt ein neues GLM-4 MoE Lite Model aus der Konfiguration
func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))

	// Erste Layers sind dense, Rest ist MoE
	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count"))
	for i := range layers {
		if i < firstDenseLayerIndex {
			layers[i].MLP = &dense{}
		} else {
			layers[i].MLP = &sparse{}
		}
	}

	// Attention Parameter berechnen
	keyLength := int(c.Uint("attention.key_length"))
	valueLength := int(c.Uint("attention.value_length"))
	kqScale := 1.0 / math.Sqrt(float64(keyLength))

	// Tokenizer Pre-Processing Pattern
	var pre []string
	switch c.String("tokenizer.ggml.pre") {
	case "glm4":
		pre = []string{
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		}
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
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
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
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

			routedScalingFactor: c.Float("expert_weights_scale"),

			kqScale: kqScale,
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

// Shift wendet RoPE-Rotation auf den KV-Cache an
func (m Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

// Validate prueft ob das Model-Format unterstuetzt wird
func (m *Model) Validate() error {
	for _, layer := range m.Layers {
		if layer.Attention != nil && (layer.Attention.KB == nil || layer.Attention.VB == nil) {
			return ErrOldModelFormat
		}
	}
	return nil
}

// Forward fuehrt einen kompletten Forward-Pass durch das Model aus
func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		// Output-Filterung nur im letzten Layer
		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	// Finale Normalisierung und Output-Projektion
	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("glm4moelite", New)
}
