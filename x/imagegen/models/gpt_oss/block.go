//go:build mlx

// block.go - Transformer Block fuer GPT-OSS
// Enthaelt Block-Struktur mit Attention und MoE.
package gpt_oss

import (
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Block represents a single transformer block
type Block struct {
	Attention    *Attention
	MLP          *MoE
	InputNorm    *nn.RMSNorm `weight:"input_layernorm"`
	PostAttnNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
	LayerType    string      // "sliding_attention" or "full_attention"
}

// Forward performs the block forward pass
func (b *Block) Forward(x *mlx.Array, c cache.Cache, B, L int32, mask *mlx.Array, maskMode string, cfg *Config) *mlx.Array {
	h := mlx.Add(x, b.Attention.Forward(b.InputNorm.Forward(x, cfg.RMSNormEps), c, B, L, mask, maskMode, cfg))
	return mlx.Add(h, b.MLP.Forward(b.PostAttnNorm.Forward(h, cfg.RMSNormEps), B, L))
}
