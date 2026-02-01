//go:build mlx

// Modul: block.go
// Beschreibung: Transformer-Block für den Qwen3 Text-Encoder.
// Enthält: Block-Struct und Forward-Methode.

package qwen3

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// Block represents a single Qwen3 transformer block
type Block struct {
	Attention         *Attention  `weight:"self_attn"`
	MLP               *MLP        `weight:"mlp"`
	InputLayerNorm    *nn.RMSNorm `weight:"input_layernorm"`
	PostAttnLayerNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
}

// Forward applies the Qwen3 block
func (qb *Block) Forward(x *mlx.Array, eps float32, mask *mlx.Array, maskMode string) *mlx.Array {
	h := qb.InputLayerNorm.Forward(x, eps)
	attnOut := qb.Attention.Forward(h, mask, maskMode)
	x = mlx.Add(x, attnOut)

	h = qb.PostAttnLayerNorm.Forward(x, eps)
	mlpOut := qb.MLP.Forward(h)
	x = mlx.Add(x, mlpOut)

	return x
}
