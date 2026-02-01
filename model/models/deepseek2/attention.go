// Modul: attention.go
// Beschreibung: Attention-Mechanismus fuer das DeepSeek2-Modell
// Hauptstrukturen:
//   - Attention: Multi-Head Attention mit MLA (Multi-Head Latent Attention) Unterstuetzung
//   - Forward: Fuehrt den Attention-Vorwaertsdurchlauf durch

package deepseek2

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// Attention implementiert den Attention-Mechanismus mit
// optionaler MLA (Multi-Head Latent Attention) Unterstuetzung
type Attention struct {
	Q *nn.Linear `gguf:"attn_q"`

	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`
	KVB     *nn.Linear  `gguf:"attn_kv_b"`

	KB *nn.Linear `gguf:"attn_k_b"`
	VB *nn.Linear `gguf:"attn_v_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
}

// Forward fuehrt den Attention-Vorwaertsdurchlauf durch.
// Unterstuetzt sowohl Standard-Attention (v3) als auch MLA (v3.1).
func (attn *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLength := hiddenStates.Dim(1)

	var query ml.Tensor
	if opts.qLoraRank == 0 {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)
	queryChunks := query.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.qkRopeHeadDim)

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kPass := compressedKV.Slice(ctx, 0, 0, opts.kvLoraRank, 1)
	kRot := compressedKV.View(ctx,
		opts.kvLoraRank*compressedKV.Stride(0), opts.qkRopeHeadDim,
		compressedKV.Stride(1), 1,
		compressedKV.Stride(1), compressedKV.Dim(1),
	)

	qRot := opts.applyRotaryPositionEmbeddings(ctx, queryChunks[1], positions)
	kRot = opts.applyRotaryPositionEmbeddings(ctx, kRot, positions)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)

	var attention ml.Tensor

	if !opts.isMLA { // v3 Standard-Attention
		kPass = attn.KVB.Forward(ctx, kPass)

		kv := kPass.Reshape(ctx, kPass.Dim(0)/opts.numKVHeads, opts.numKVHeads, seqLength)
		kvChunks := kv.ChunkSections(ctx, 0, opts.kqNopeHeadDim, opts.vHeadDim)

		kRot = kRot.Repeat(ctx, 1, queryChunks[0].Dim(1))
		query = qRot.Concat(ctx, queryChunks[0], 0)
		key := kRot.Concat(ctx, kvChunks[0], 0)
		attention = nn.Attention(ctx, query, key, kvChunks[1], opts.kqScale, cache)
	} else { // v3.1 MLA (Multi-Head Latent Attention)
		qPass := queryChunks[0].Permute(ctx, 0, 2, 1, 3)
		qPassAbsorb := attn.KB.Forward(ctx, qPass)
		qPassAbsorb = qPassAbsorb.Permute(ctx, 0, 2, 1, 3)

		query = qRot.Concat(ctx, qPassAbsorb, 0)
		kPass = kPass.Reshape(ctx, opts.kvLoraRank, 1, seqLength)
		key := kRot.Concat(ctx, kPass, 0)
		value := kPass

		attention = nn.AttentionWithVMLA(ctx, query, key, value, nil, attn.VB.Weight, opts.kqScale, cache)
	}

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention)
}
