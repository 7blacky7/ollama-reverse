//go:build mlx

// moe.go - Mixture of Experts Layer fuer GPT-OSS
// Enthaelt MoE-Struktur und SwiGLU-Aktivierungsfunktion.
package gpt_oss

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// swiGLU applies the GPT-OSS custom SwiGLU activation.
// Formula: (gate * sigmoid(alpha * gate)) * (up + 1)
// with clipping: gate to [None, limit], up to [-limit, limit]
func swiGLU(gate, up *mlx.Array, alpha, limit float32) *mlx.Array {
	// Clip gate to [None, limit]
	gateClipped := mlx.ClipScalar(gate, 0, limit, false, true)

	// Clip up to [-limit, limit]
	upClipped := mlx.ClipScalar(up, -limit, limit, true, true)

	// glu_scaled = alpha * gate_clipped
	gluScaled := mlx.MulScalar(gateClipped, alpha)

	// sig = sigmoid(glu_scaled)
	sig := mlx.Sigmoid(gluScaled)

	// out_glu = gate_clipped * sig
	outGlu := mlx.Mul(gateClipped, sig)

	// result = out_glu * (up_clipped + 1)
	return mlx.Mul(outGlu, mlx.AddScalar(upClipped, 1.0))
}

// compiledSwiGLU is a singleton compiled SwiGLU function shared across all layers
var compiledSwiGLU *mlx.CompiledFunc

// getCompiledSwiGLU returns the compiled SwiGLU function, creating it once if needed
func getCompiledSwiGLU() *mlx.CompiledFunc {
	if compiledSwiGLU == nil {
		const alpha float32 = 1.702
		const limit float32 = 7.0
		compiledSwiGLU = mlx.CompileShapeless(func(inputs []*mlx.Array) []*mlx.Array {
			return []*mlx.Array{swiGLU(inputs[0], inputs[1], alpha, limit)}
		}, true)
	}
	return compiledSwiGLU
}

// MoE represents the Mixture of Experts SwiGLU layer with quantized experts.
type MoE struct {
	Router     *nn.Linear `weight:"mlp.router"`
	TopK       int32
	HiddenSize int32
	GroupSize  int
	Bits       int
	// Expert weights (loaded manually via sanitizeExpertWeights)
	GateBlocks, GateScales, GateBias *mlx.Array
	UpBlocks, UpScales, UpBias       *mlx.Array
	DownBlocks, DownScales, DownBias *mlx.Array
}

// Forward performs the MoE forward pass
func (moe *MoE) Forward(x *mlx.Array, B, L int32) *mlx.Array {
	logits := moe.Router.Forward(x)
	neg := mlx.Neg(logits)
	part := mlx.Argpartition(neg, int(moe.TopK)-1, -1)
	topKIdx := mlx.Slice(part, []int32{0, 0, 0}, []int32{B, L, moe.TopK})
	topKVal := mlx.TakeAlongAxis(logits, topKIdx, -1)
	weights := mlx.Softmax(topKVal, -1)

	xFlat := mlx.Reshape(x, B*L, 1, 1, moe.HiddenSize)
	idxFlat := mlx.Reshape(topKIdx, B*L, moe.TopK)

	doSort := B*L >= 64
	var invOrder *mlx.Array
	sorted := false
	n := B * L * moe.TopK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, moe.TopK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
		sorted = true
	}

	gate := mlx.GatherQMM(xFlat, moe.GateBlocks, moe.GateScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)
	up := mlx.GatherQMM(xFlat, moe.UpBlocks, moe.UpScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)

	if moe.GateBias != nil {
		gate = mlx.Add(gate, mlx.ExpandDims(mlx.Take(moe.GateBias, idxFlat, 0), 2))
	}
	if moe.UpBias != nil {
		up = mlx.Add(up, mlx.ExpandDims(mlx.Take(moe.UpBias, idxFlat, 0), 2))
	}

	hidden := getCompiledSwiGLU().Call(gate, up)[0]

	down := mlx.GatherQMM(hidden, moe.DownBlocks, moe.DownScales, nil, nil, idxFlat, true, moe.GroupSize, moe.Bits, "mxfp4", sorted)
	if moe.DownBias != nil {
		down = mlx.Add(down, mlx.ExpandDims(mlx.Take(moe.DownBias, idxFlat, 0), 2))
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, moe.TopK, moe.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	ewFlat := mlx.Reshape(weights, B*L, moe.TopK, 1)
	return mlx.Reshape(mlx.Sum(mlx.Mul(down, ewFlat), 1, false), B, L, moe.HiddenSize)
}

// sanitizeExpertWeights splits merged gate_up weights into separate gate/up arrays.
// MXFP4 quantized weights require contiguous memory - strided views give wrong results.
func sanitizeExpertWeights(weights *safetensors.ModelWeights, prefix string) (moe *MoE) {
	gateUpBlocks, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_blocks")
	gateUpScales, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_scales")
	gateUpBias, _ := weights.GetTensor(prefix + ".mlp.experts.gate_up_proj_bias")
	downBlocks, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_blocks")
	downScales, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_scales")
	downBias, _ := weights.GetTensor(prefix + ".mlp.experts.down_proj_bias")

	moe = &MoE{GroupSize: 32, Bits: 4, DownScales: downScales, DownBias: downBias}

	if gateUpBlocks != nil {
		gub := mlx.FlattenRange(mlx.View(gateUpBlocks, int(mlx.DtypeUint32)), -2, -1)
		s := gub.Shape()
		moe.GateBlocks = mlx.Contiguous(mlx.SliceStride(gub, []int32{0, 0, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
		moe.UpBlocks = mlx.Contiguous(mlx.SliceStride(gub, []int32{0, 1, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
	}
	if gateUpScales != nil {
		s := gateUpScales.Shape()
		moe.GateScales = mlx.Contiguous(mlx.SliceStride(gateUpScales, []int32{0, 0, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
		moe.UpScales = mlx.Contiguous(mlx.SliceStride(gateUpScales, []int32{0, 1, 0}, []int32{s[0], s[1], s[2]}, []int32{1, 2, 1}))
	}
	if gateUpBias != nil {
		s := gateUpBias.Shape()
		moe.GateBias = mlx.Contiguous(mlx.SliceStride(gateUpBias, []int32{0, 0}, []int32{s[0], s[1]}, []int32{1, 2}))
		moe.UpBias = mlx.Contiguous(mlx.SliceStride(gateUpBias, []int32{0, 1}, []int32{s[0], s[1]}, []int32{1, 2}))
	}
	if downBlocks != nil {
		moe.DownBlocks = mlx.FlattenRange(mlx.View(downBlocks, int(mlx.DtypeUint32)), -2, -1)
	}
	return moe
}
