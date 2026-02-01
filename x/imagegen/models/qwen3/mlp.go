//go:build mlx

// Modul: mlp.go
// Beschreibung: SwiGLU MLP-Implementierung für den Qwen3 Text-Encoder.
// Enthält: MLP-Struct und Forward-Methode.

package qwen3

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// MLP implements Qwen3 SwiGLU MLP
type MLP struct {
	GateProj nn.LinearLayer `weight:"gate_proj"`
	UpProj   nn.LinearLayer `weight:"up_proj"`
	DownProj nn.LinearLayer `weight:"down_proj"`
}

// Forward applies the MLP
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	gate = mlx.SiLU(gate)
	up := m.UpProj.Forward(x)
	h := mlx.Mul(gate, up)
	return m.DownProj.Forward(h)
}
