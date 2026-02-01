// Package ggml - Graph Size Berechnung
//
// Dieses Modul enthaelt Methoden zur Berechnung des GPU-Speicherbedarfs:
// - GraphSize: Berechnet KV-Cache und Offload-Groessen pro Architektur
// - SupportsKVCacheType: Prueft unterstuetzte Cache-Typen
// - SupportsFlashAttention: Prueft Flash-Attention Unterstuetzung
// - FlashAttention: Prueft ob Flash-Attention aktiviert werden soll
package ggml

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// GraphSize berechnet den KV-Cache und Offload-Speicherbedarf
// Gibt zurueck: kv (pro Layer), partialOffload, fullOffload in Bytes
func (f GGML) GraphSize(context, batch uint64, numParallel int, kvCacheType string, useFlashAttention ml.FlashAttentionType) (kv []uint64, partialOffload, fullOffload uint64) {
	context *= uint64(numParallel)

	embedding := f.KV().EmbeddingLength()
	heads := f.KV().HeadCountMax()
	headsArr := f.KV().HeadCount()
	headsKV := f.KV().HeadCountKVMax()
	headsKVArr := f.KV().HeadCountKV()
	vocab := uint64(f.KV()["tokenizer.ggml.tokens"].(*array[string]).size)

	embeddingHeads := f.KV().EmbeddingHeadCountMax()
	embeddingHeadsK := f.KV().EmbeddingHeadCountK()
	embeddingHeadsV := f.KV().EmbeddingHeadCountV()

	layers := f.Tensors().GroupLayers()
	bytesPerElement := kvCacheBytesPerElement(kvCacheType)

	// Default KV-Cache Berechnung fuer alle Modelle
	var kvTotal uint64
	kv = make([]uint64, f.KV().BlockCount())
	kvSizeAttn := uint64(0)
	kvSizeRecurrent := uint64(0)

	for i := range kv {
		headsL := headsArr[i]
		headsKVL := headsKVArr[i]
		if headsL > 0 && headsKVL > 0 {
			// Attention Layer
			kv[i] = uint64(float64(context*(embeddingHeadsK+embeddingHeadsV)*headsKVL) * bytesPerElement)
			kvSizeAttn += kv[i]
		} else {
			// Recurrent Layer (SSM)
			ssmDConv := f.KV().SSMConvKernel()
			ssmDState := f.KV().SSMStateSize()
			ssmDInner := f.KV().SSMInnerSize()
			ssmNGroups := f.KV().SSMGroupCount()
			nEmbdR := uint64(0)
			if ssmDConv > 0 {
				nEmbdR = (ssmDConv - 1) * (ssmDInner + 2*ssmNGroups*ssmDState)
			}
			nEmbdS := ssmDState * ssmDInner
			bytesPerElementRecurrent := kvCacheBytesPerElement("f32")
			kv[i] = (nEmbdR + nEmbdS) * uint64(bytesPerElementRecurrent)
			kvSizeRecurrent += kv[i]
		}
		kvTotal += kv[i]
	}
	slog.Debug("default cache size estimate", "attention MiB", float32(kvSizeAttn)/(1024.*1024.), "attention bytes", kvSizeAttn, "recurrent MiB", float32(kvSizeRecurrent)/(1024.*1024.), "recurrent bytes", kvSizeRecurrent)

	// Architektur-spezifische Berechnungen
	switch f.KV().Architecture() {
	case "llama", "llama4":
		fullOffload, partialOffload = f.graphSizeLlama(batch, embedding, context, heads, headsKV, vocab, embeddingHeads, layers)
	case "mllama":
		kv, fullOffload, partialOffload = f.graphSizeMllama(kv, batch, embedding, context, heads, headsKV, vocab, embeddingHeadsK, embeddingHeadsV, bytesPerElement)
	case "gemma", "gemma2", "gemma3", "gemma3n":
		kv, fullOffload, partialOffload = f.graphSizeGemma(kv, batch, embedding, context, heads, headsKV, vocab, embeddingHeadsK, embeddingHeadsV, bytesPerElement, numParallel)
	case "command-r":
		fullOffload, partialOffload = f.graphSizeCommandR(batch, embedding, context, heads, vocab)
	case "qwen2":
		fullOffload, partialOffload = f.graphSizeQwen2(batch, embedding, context, heads, vocab)
	case "phi2":
		fullOffload, partialOffload = f.graphSizePhi2(batch, embedding, context, heads, vocab)
	case "stablelm":
		fullOffload, partialOffload = f.graphSizeStableLM(batch, embedding, context, heads, vocab)
	case "deepseek2":
		fullOffload, partialOffload = f.graphSizeDeepseek2(batch, embedding, context, headsKV, vocab, embeddingHeadsK)
	case "chatglm":
		fullOffload, partialOffload = f.graphSizeChatGLM(batch, embedding, context, heads, vocab, embeddingHeadsK, layers)
	case "gptoss", "gpt-oss":
		kv, partialOffload = f.graphSizeGptOss(batch, context, headsKV, embeddingHeadsK, embeddingHeadsV, bytesPerElement, kvTotal, numParallel, useFlashAttention)
	}

	return
}

// graphSizeLlama berechnet Speicherbedarf fuer Llama-Modelle
func (f GGML) graphSizeLlama(batch, embedding, context, heads, headsKV, vocab, embeddingHeads uint64, layers map[string]Layer) (fullOffload, partialOffload uint64) {
	fullOffload = max(
		4*batch*(1+4*embedding+context*(1+heads)),
		4*batch*(embedding+vocab),
	)

	partialOffload = 4 * batch * embedding
	partialOffload += max(
		4*batch*(1+embedding+max(context, embedding))+embedding*embedding*9/16+4*context*(batch*heads+embeddingHeads*headsKV),
		4*batch*(embedding+vocab)+embedding*vocab*105/128,
	)

	if ffnGateExpsWeight, ok := layers["blk.0"]["ffn_gate_exps.weight"]; ok {
		ff := uint64(f.KV().Uint("feed_forward_length"))
		partialOffload = max(
			3*ffnGateExpsWeight.Size()+4*batch*(2*ff+headsKV+embedding+context+embeddingHeads*headsKV),
			4*(context*batch*heads+context*embeddingHeads*headsKV+batch*1024+embeddingHeads*headsKV*batch),
		)
	} else if ffnGateWeight, ok := layers["blk.0"]["ffn_gate.0.weight"]; ok {
		ffnGateWeight1 := ffnGateWeight.Shape[1]
		fullOffload = 4 * batch * (2 + 3*embedding + context*(1+heads) + 2*headsKV + ffnGateWeight1)
		partialOffload = max(
			4*batch*(3+embeddingHeads*headsKV+embedding+context*(1+heads)+ffnGateWeight1)+(embedding*embedding+3*embedding*headsKV*ffnGateWeight1)*9/16,
			4*batch*(1+2*embedding+context*(1+heads))+embedding*(6*context*headsKV/heads+embedding*9/16),
		)
	}
	return
}

// graphSizeMllama berechnet Speicherbedarf fuer MLlama-Modelle
func (f GGML) graphSizeMllama(kv []uint64, batch, embedding, context, heads, headsKV, vocab, embeddingHeadsK, embeddingHeadsV uint64, bytesPerElement float64) ([]uint64, uint64, uint64) {
	var visionTokens, tiles uint64 = 1601, 4

	crossAttentionLayers := f.KV().Ints("attention.cross_attention_layers")
	for i := range kv {
		if slices.Contains(crossAttentionLayers, int32(i)) {
			kv[i] = headsKV * (embeddingHeadsK + embeddingHeadsV) * 4 * visionTokens * tiles
		}
	}

	fullOffload := max(
		4*batch*(2+3*embedding+embeddingHeadsK*heads+context*(1+heads)),
		4*batch*(embedding+vocab),
	)

	var ropeFreqsCount uint64
	if ropeFreqs, ok := f.Tensors().GroupLayers()["rope_freqs"]; ok {
		if ropeFreqsWeights, ok := ropeFreqs["weights"]; ok {
			ropeFreqsCount = ropeFreqsWeights.Elements()
		}
	}

	partialOffload := max(
		4*(batch*(2*embedding+1+context*(1+heads)+embeddingHeadsK*heads)+ropeFreqsCount+embeddingHeadsK*context*headsKV),
		4*batch*(embedding+vocab)+embedding*vocab*105/128,
	)

	return kv, fullOffload, partialOffload
}

// graphSizeGemma berechnet Speicherbedarf fuer Gemma-Modelle
func (f GGML) graphSizeGemma(kv []uint64, batch, embedding, context, heads, headsKV, vocab, embeddingHeadsK, embeddingHeadsV uint64, bytesPerElement float64, numParallel int) ([]uint64, uint64, uint64) {
	fullOffload := max(
		4*batch*(embedding+vocab),
		4*batch*(2+context+context*heads+2*embedding+2*embeddingHeadsK*heads),
	)

	partialOffload := max(
		4*embedding*batch+embedding*vocab*105/128+4*vocab*batch,
		4*batch*(2*embedding+1+2*embeddingHeadsK*heads+context+context*heads)+4*embeddingHeadsK*context*8+embedding*embeddingHeadsK*heads*9/16,
	)

	if f.KV().Architecture() == "gemma3n" {
		fullOffload *= 4
		partialOffload *= 4
	}

	if f.KV().Architecture() == "gemma3" {
		const gemma3GlobalCacheCount = 6
		slidingWindow := (uint64(numParallel) * uint64(f.KV().Uint("attention.sliding_window"))) + batch
		for i := range kv {
			if (i+1)%gemma3GlobalCacheCount != 0 {
				kv[i] = uint64(float64(slidingWindow*(embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)
			}
		}
	}

	return kv, fullOffload, partialOffload
}

// graphSizeCommandR berechnet Speicherbedarf fuer Command-R
func (f GGML) graphSizeCommandR(batch, embedding, context, heads, vocab uint64) (fullOffload, partialOffload uint64) {
	fullOffload = max(4*batch*(embedding+vocab), 4*batch*(2+4*embedding+context*(1+heads)))
	partialOffload = max(4*batch*(embedding+vocab)+embedding*vocab*105/128, 4*batch*(1+2*embedding+context*(1+heads))+4*embedding*context+embedding*embedding*9/16)
	return
}

// graphSizeQwen2 berechnet Speicherbedarf fuer Qwen2
func (f GGML) graphSizeQwen2(batch, embedding, context, heads, vocab uint64) (fullOffload, partialOffload uint64) {
	fullOffload = max(4*batch*(embedding+vocab), 4*batch*(1+2*embedding+context+context*heads))
	partialOffload = max(4*batch*(embedding+vocab)+embedding*vocab*105/128, 4*(batch*(1+2*embedding+context*(1+heads))+embedding*(1+context)))
	return
}

// graphSizePhi2 berechnet Speicherbedarf fuer Phi2
func (f GGML) graphSizePhi2(batch, embedding, context, heads, vocab uint64) (fullOffload, partialOffload uint64) {
	fullOffload = max(4*batch*(embedding+vocab), 4*batch*(1+4*embedding+context+context*heads))
	partialOffload = max(4*batch*(2*embedding+vocab)+embedding*vocab*105/128, 4*batch*(2+3*embedding+context+context*heads))
	return
}

// graphSizeStableLM berechnet Speicherbedarf fuer StableLM
func (f GGML) graphSizeStableLM(batch, embedding, context, heads, vocab uint64) (fullOffload, partialOffload uint64) {
	fullOffload = 4 * batch * (context*(1+heads) + 3*embedding + 2)
	partialOffload = max(4*batch*(vocab+2*embedding), fullOffload)
	return
}

// graphSizeDeepseek2 berechnet Speicherbedarf fuer Deepseek2
func (f GGML) graphSizeDeepseek2(batch, embedding, context, headsKV, vocab, embeddingHeadsK uint64) (fullOffload, partialOffload uint64) {
	fullOffload = max(4*batch*(3*embedding+vocab), 4*batch*(3*embedding+2+context*(1+headsKV)+2*embeddingHeadsK*headsKV))
	partialOffload = max(4*batch*(3*embedding+vocab)+embedding*vocab*105/128, 4*batch*(2*embedding+1+2*embeddingHeadsK*headsKV+context+context*headsKV)+4*embeddingHeadsK*context*headsKV+embedding*embeddingHeadsK*headsKV*9/16)
	return
}

// graphSizeChatGLM berechnet Speicherbedarf fuer ChatGLM
func (f GGML) graphSizeChatGLM(batch, embedding, context, heads, vocab, embeddingHeadsK uint64, layers map[string]Layer) (fullOffload, partialOffload uint64) {
	fullOffload = 4 * batch * (embedding + vocab)
	partialOffload = 4*batch*(embedding+vocab) + embedding*vocab*105/128
	if qkvBias, ok := layers["blk.0"]["attn_qkv.bias"]; ok {
		fullOffload = max(fullOffload, 4*batch*(2+2*embedding+context+context*heads+embeddingHeadsK*heads+qkvBias.Shape[0]))
		partialOffload = max(partialOffload, 4*batch*(1+2*embedding+embeddingHeadsK*heads+context+context*heads)+4*embeddingHeadsK*context+4*context*embeddingHeadsK+4*qkvBias.Shape[0])
	}
	return
}

// graphSizeGptOss berechnet Speicherbedarf fuer GPT-OSS
func (f GGML) graphSizeGptOss(batch, context, headsKV, embeddingHeadsK, embeddingHeadsV uint64, bytesPerElement float64, kvTotal uint64, numParallel int, useFlashAttention ml.FlashAttentionType) ([]uint64, uint64) {
	kv := make([]uint64, f.KV().BlockCount())
	for i := range kv {
		kv[i] = uint64(float64((embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)
		if i%2 == 0 {
			kv[i] *= (uint64(numParallel)*4096 + batch)
		} else {
			kv[i] *= context
		}
	}

	partialOffload := 2 * f.KV().HeadCountMax() / cmp.Or(f.KV().HeadCountKVMin(), 1) * kvTotal / 6
	if useFlashAttention == ml.FlashAttentionEnabled {
		partialOffload = (4*uint64(numParallel) + context>>10 + 110) * format.MebiByte
	}
	return kv, partialOffload
}

// SupportsKVCacheType prueft ob ein Cache-Typ unterstuetzt wird
func (f GGML) SupportsKVCacheType(cacheType string) bool {
	if cacheType == "" || cacheType == "f16" {
		return true
	}
	return slices.Contains([]string{"q8_0", "q4_0"}, cacheType)
}

// KVCacheTypeIsQuantized prueft ob ein Cache-Typ quantisiert ist
func (f GGML) KVCacheTypeIsQuantized(cacheType string) bool {
	if cacheType == "" || cacheType == "f16" || cacheType == "f32" || cacheType == "bf16" {
		return false
	}
	return true
}

// SupportsFlashAttention prueft ob Flash-Attention unterstuetzt wird
func (f GGML) SupportsFlashAttention() bool {
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", f.KV().Architecture())]
	if isEmbedding {
		return false
	}

	if arch := f.KV().Architecture(); slices.Contains([]string{"gemma2"}, arch) {
		return false
	}

	headCountK := f.KV().EmbeddingHeadCountK()
	headCountV := f.KV().EmbeddingHeadCountV()
	return headCountK != 0 && headCountV != 0 && headCountK == headCountV
}

// FlashAttention prueft ob Flash-Attention aktiviert werden soll
func (f GGML) FlashAttention() bool {
	return slices.Contains([]string{
		"bert",
		"gemma3",
		"glm4moelite",
		"gptoss", "gpt-oss",
		"lfm2",
		"mistral3",
		"olmo3",
		"qwen3", "qwen3moe",
		"qwen3vl", "qwen3vlmoe",
	}, f.KV().String("general.architecture"))
}

// kvCacheBytesPerElement gibt die Bytes pro Element fuer einen Cache-Typ zurueck
func kvCacheBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 1 // 1/2 of fp16
	case "q4_0":
		return 0.5 // 1/4 of fp16
	case "f32":
		return 4 // f32 (default for recurrent)
	default:
		return 2 // f16 (default)
	}
}
