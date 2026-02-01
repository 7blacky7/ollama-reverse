//go:build mlx

// Package flux2 - Transformer-Bloecke
// Enthält: TransformerBlock (dual-stream) und SingleTransformerBlock

package flux2

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// TransformerBlockAttn implements dual-stream attention
// Weight names: transformer_blocks.N.attn.*
type TransformerBlockAttn struct {
	// Image stream (separate Q, K, V projections)
	ToQ nn.LinearLayer `weight:"to_q"`
	ToK nn.LinearLayer `weight:"to_k"`
	ToV nn.LinearLayer `weight:"to_v"`
	// Note: to_out has .0 suffix in weights, handled specially
	ToOut0 nn.LinearLayer `weight:"to_out.0"`

	// Text stream (add_ projections)
	AddQProj nn.LinearLayer `weight:"add_q_proj"`
	AddKProj nn.LinearLayer `weight:"add_k_proj"`
	AddVProj nn.LinearLayer `weight:"add_v_proj"`
	ToAddOut nn.LinearLayer `weight:"to_add_out"`

	// QK norms for image stream
	NormQ *mlx.Array `weight:"norm_q.weight"`
	NormK *mlx.Array `weight:"norm_k.weight"`

	// QK norms for text stream (added)
	NormAddedQ *mlx.Array `weight:"norm_added_q.weight"`
	NormAddedK *mlx.Array `weight:"norm_added_k.weight"`
}

// TransformerBlock implements a dual-stream transformer block
// Weight names: transformer_blocks.N.*
type TransformerBlock struct {
	Attn      *TransformerBlockAttn `weight:"attn"`
	FF        *FeedForward          `weight:"ff"`
	FFContext *FeedForward          `weight:"ff_context"`

	// Config (set after loading)
	NHeads  int32
	HeadDim int32
	Scale   float32
}

// Forward applies the dual-stream block
// imgHidden: [B, imgLen, dim]
// txtHidden: [B, txtLen, dim]
// imgMod, txtMod: modulation params [B, 6*dim] each
// cos, sin: RoPE values
func (block *TransformerBlock) Forward(imgHidden, txtHidden *mlx.Array, imgMod, txtMod *mlx.Array, cos, sin *mlx.Array) (*mlx.Array, *mlx.Array) {
	imgShape := imgHidden.Shape()
	B := imgShape[0]
	imgLen := imgShape[1]
	dim := imgShape[2]
	txtLen := txtHidden.Shape()[1]

	// Parse modulation: 6 params each (shift1, scale1, gate1, shift2, scale2, gate2)
	imgShift1, imgScale1, imgGate1 := parseModulation3(imgMod, dim, 0)
	imgShift2, imgScale2, imgGate2 := parseModulation3(imgMod, dim, 3)
	txtShift1, txtScale1, txtGate1 := parseModulation3(txtMod, dim, 0)
	txtShift2, txtScale2, txtGate2 := parseModulation3(txtMod, dim, 3)

	// === Attention branch ===
	// Modulate inputs
	imgNorm := modulateLayerNorm(imgHidden, imgShift1, imgScale1)
	txtNorm := modulateLayerNorm(txtHidden, txtShift1, txtScale1)

	// Compute Q, K, V for image stream (separate projections)
	imgQ := block.Attn.ToQ.Forward(imgNorm)
	imgK := block.Attn.ToK.Forward(imgNorm)
	imgV := block.Attn.ToV.Forward(imgNorm)

	// Compute Q, K, V for text stream (add_ projections)
	txtQ := block.Attn.AddQProj.Forward(txtNorm)
	txtK := block.Attn.AddKProj.Forward(txtNorm)
	txtV := block.Attn.AddVProj.Forward(txtNorm)

	// Reshape for attention: [B, L, dim] -> [B, L, nheads, headDim]
	imgQ = mlx.Reshape(imgQ, B, imgLen, block.NHeads, block.HeadDim)
	imgK = mlx.Reshape(imgK, B, imgLen, block.NHeads, block.HeadDim)
	imgV = mlx.Reshape(imgV, B, imgLen, block.NHeads, block.HeadDim)
	txtQ = mlx.Reshape(txtQ, B, txtLen, block.NHeads, block.HeadDim)
	txtK = mlx.Reshape(txtK, B, txtLen, block.NHeads, block.HeadDim)
	txtV = mlx.Reshape(txtV, B, txtLen, block.NHeads, block.HeadDim)

	// Apply QK norm (RMSNorm with learned scale)
	imgQ = applyQKNorm(imgQ, block.Attn.NormQ)
	imgK = applyQKNorm(imgK, block.Attn.NormK)
	txtQ = applyQKNorm(txtQ, block.Attn.NormAddedQ)
	txtK = applyQKNorm(txtK, block.Attn.NormAddedK)

	// Concatenate for joint attention: text first, then image
	q := mlx.Concatenate([]*mlx.Array{txtQ, imgQ}, 1)
	k := mlx.Concatenate([]*mlx.Array{txtK, imgK}, 1)
	v := mlx.Concatenate([]*mlx.Array{txtV, imgV}, 1)

	// Apply RoPE
	q = ApplyRoPE4D(q, cos, sin)
	k = ApplyRoPE4D(k, cos, sin)

	// Transpose for SDPA: [B, nheads, L, headDim]
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Scaled dot-product attention
	out := mlx.ScaledDotProductAttention(q, k, v, block.Scale, false)

	// Transpose back: [B, L, nheads, headDim]
	out = mlx.Transpose(out, 0, 2, 1, 3)

	// Split back into txt and img
	totalLen := txtLen + imgLen
	txtOut := mlx.Slice(out, []int32{0, 0, 0, 0}, []int32{B, txtLen, block.NHeads, block.HeadDim})
	imgOut := mlx.Slice(out, []int32{0, txtLen, 0, 0}, []int32{B, totalLen, block.NHeads, block.HeadDim})

	// Reshape and project
	txtOut = mlx.Reshape(txtOut, B, txtLen, dim)
	imgOut = mlx.Reshape(imgOut, B, imgLen, dim)
	txtOut = block.Attn.ToAddOut.Forward(txtOut)
	imgOut = block.Attn.ToOut0.Forward(imgOut)

	// Apply gates and residual
	imgHidden = mlx.Add(imgHidden, mlx.Mul(imgGate1, imgOut))
	txtHidden = mlx.Add(txtHidden, mlx.Mul(txtGate1, txtOut))

	// === MLP branch ===
	imgNorm = modulateLayerNorm(imgHidden, imgShift2, imgScale2)
	txtNorm = modulateLayerNorm(txtHidden, txtShift2, txtScale2)

	imgFFOut := block.FF.Forward(imgNorm)
	txtFFOut := block.FFContext.Forward(txtNorm)

	imgHidden = mlx.Add(imgHidden, mlx.Mul(imgGate2, imgFFOut))
	txtHidden = mlx.Add(txtHidden, mlx.Mul(txtGate2, txtFFOut))

	return imgHidden, txtHidden
}

// SingleTransformerBlockAttn implements attention for single-stream blocks
// Weight names: single_transformer_blocks.N.attn.*
type SingleTransformerBlockAttn struct {
	ToQKVMlpProj nn.LinearLayer `weight:"to_qkv_mlp_proj"` // Fused QKV + MLP input
	ToOut        nn.LinearLayer `weight:"to_out"`          // Fused attn_out + MLP out
	NormQ        *mlx.Array     `weight:"norm_q.weight"`
	NormK        *mlx.Array     `weight:"norm_k.weight"`
}

// SingleTransformerBlock implements a single-stream transformer block
// Weight names: single_transformer_blocks.N.*
type SingleTransformerBlock struct {
	Attn *SingleTransformerBlockAttn `weight:"attn"`

	// Config
	NHeads    int32
	HeadDim   int32
	InnerDim  int32
	MLPHidDim int32
	Scale     float32
}

// Forward applies the single-stream block
// x: [B, L, dim] concatenated text+image
// mod: modulation [B, 3*dim]
func (block *SingleTransformerBlock) Forward(x *mlx.Array, mod *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	dim := shape[2]

	// Parse modulation: (shift, scale, gate)
	shift, scale, gate := parseModulation3(mod, dim, 0)

	// Modulate input
	h := modulateLayerNorm(x, shift, scale)

	// Fused projection: QKV + MLP gate/up
	// linear1 outputs: [q, k, v, mlp_gate, mlp_up] = [dim, dim, dim, mlpHid, mlpHid]
	qkvMlp := block.Attn.ToQKVMlpProj.Forward(h)

	// Split: first 3*dim is QKV, rest is MLP
	qkvDim := 3 * block.InnerDim
	qkv := mlx.Slice(qkvMlp, []int32{0, 0, 0}, []int32{B, L, qkvDim})
	mlpIn := mlx.Slice(qkvMlp, []int32{0, 0, qkvDim}, []int32{B, L, qkvMlp.Shape()[2]})

	// Split QKV
	q, k, v := splitQKV(qkv, B, L, block.InnerDim)

	// Reshape for attention
	q = mlx.Reshape(q, B, L, block.NHeads, block.HeadDim)
	k = mlx.Reshape(k, B, L, block.NHeads, block.HeadDim)
	v = mlx.Reshape(v, B, L, block.NHeads, block.HeadDim)

	// QK norm
	q = applyQKNorm(q, block.Attn.NormQ)
	k = applyQKNorm(k, block.Attn.NormK)

	// Apply RoPE
	q = ApplyRoPE4D(q, cos, sin)
	k = ApplyRoPE4D(k, cos, sin)

	// Transpose for SDPA
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// SDPA
	attnOut := mlx.ScaledDotProductAttention(q, k, v, block.Scale, false)

	// Transpose back and reshape
	attnOut = mlx.Transpose(attnOut, 0, 2, 1, 3)
	attnOut = mlx.Reshape(attnOut, B, L, block.InnerDim)

	// MLP: SwiGLU
	mlpShape := mlpIn.Shape()
	half := mlpShape[2] / 2
	mlpGate := mlx.Slice(mlpIn, []int32{0, 0, 0}, []int32{B, L, half})
	mlpUp := mlx.Slice(mlpIn, []int32{0, 0, half}, []int32{B, L, mlpShape[2]})
	mlpOut := mlx.Mul(mlx.SiLU(mlpGate), mlpUp)

	// Concatenate attention and MLP for fused output
	combined := mlx.Concatenate([]*mlx.Array{attnOut, mlpOut}, 2)

	// Output projection
	out := block.Attn.ToOut.Forward(combined)

	// Apply gate and residual
	return mlx.Add(x, mlx.Mul(gate, out))
}
