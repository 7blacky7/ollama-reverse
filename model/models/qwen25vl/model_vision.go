package qwen25vl

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

// ============================================================================
// Vision Model - Hauptmodell fuer Qwen 2.5 VL Vision-Verarbeitung
// ============================================================================
//
// Dieses Modul enthaelt:
// - VisionModelOptions: Konfigurationsoptionen fuer das Vision-Modell
// - VisionModel: Hauptstruktur mit Encoder-Layers und Forward-Pass
// - blockDiagonalMask: Attention-Maske fuer Block-Diagonale Attention
// - windowIndex: Fenster-Indexierung fuer effiziente Verarbeitung

// VisionModelOptions enthaelt Konfigurationsoptionen fuer das Vision-Modell
type VisionModelOptions struct {
	hiddenSize        int
	numHeads          int
	headDim           int
	patchSize         int
	numChannels       int
	eps               float32
	ropeTheta         float32
	spatialMergeSize  int
	windowSize        int
	fullAttnBlocks    []int32
	temporalPatchSize int
}

// applyRotaryPositionEmbeddings wendet Rotary Position Embeddings an
func (o VisionModelOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	// RoPE mit 4-facher Aufteilung der Head-Dimension
	quarterHeadDim := o.headDim / 4
	return nn.RoPE(ctx, states, positions, o.headDim/2, o.ropeTheta, 1,
		rope.WithVision([]int{
			quarterHeadDim,
			quarterHeadDim,
			quarterHeadDim,
			quarterHeadDim,
		}),
	)
}

// VisionModel implementiert das Qwen Vision-Modell
type VisionModel struct {
	PatchEmbedding *PatchEmbedding
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PatchMerger    *VisionPatchMerger   `gguf:"merger"`

	*VisionModelOptions
}

// Forward berechnet das Vision-Modell fuer einen Eingabe-Tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) ml.Tensor {
	// Patch-Embeddings extrahieren
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.VisionModelOptions)

	index, bounds := m.windowIndex(grid)
	spatialMergeUnit := m.spatialMergeSize * m.spatialMergeSize

	// Fenster-Index anwenden
	windowIndex := ctx.Input().FromInts(index, len(index))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)*spatialMergeUnit, hiddenStates.Dim(1)/spatialMergeUnit)
	hiddenStates = hiddenStates.Rows(ctx, windowIndex.Argsort(ctx))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)/spatialMergeUnit, hiddenStates.Dim(1)*spatialMergeUnit)

	// Positionen berechnen
	positions := m.computePositions(ctx, grid, index, spatialMergeUnit)

	// Attention-Maske erstellen
	mask := blockDiagonalMask(ctx, hiddenStates.Dim(1), bounds)

	// Encoder-Layers anwenden
	for i, layer := range m.Layers {
		if slices.Contains(m.fullAttnBlocks, int32(i)) {
			// Full Attention (ohne Maske)
			hiddenStates = layer.Forward(ctx, hiddenStates, positions, nil, m.VisionModelOptions)
		} else {
			// Window Attention (mit Maske)
			hiddenStates = layer.Forward(ctx, hiddenStates, positions, mask, m.VisionModelOptions)
		}
	}

	// Patch-Merging und finale Sortierung
	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, m.VisionModelOptions)
	return hiddenStates.Rows(ctx, windowIndex)
}

// computePositions berechnet die Positionen fuer RoPE
func (m *VisionModel) computePositions(ctx ml.Context, grid *Grid, index []int32, spatialMergeUnit int) ml.Tensor {
	// 4 Positions-Arrays fuer x/y Koordinaten (jeweils doppelt fuer RoPE)
	s := [][]int32{
		make([]int32, grid.Height*grid.Width),
		make([]int32, grid.Height*grid.Width),
		make([]int32, grid.Height*grid.Width),
		make([]int32, grid.Height*grid.Width),
	}

	var cur int
	for y := 0; y < grid.Height; y += m.spatialMergeSize {
		for x := 0; x < grid.Width; x += m.spatialMergeSize {
			for dy := range 2 {
				for dx := range 2 {
					i := int(index[cur/spatialMergeUnit]) * spatialMergeUnit
					i += cur % spatialMergeUnit
					s[0][i] = int32(y + dy)
					s[1][i] = int32(x + dx)
					s[2][i] = int32(y + dy)
					s[3][i] = int32(x + dx)
					cur++
				}
			}
		}
	}

	return ctx.Input().FromInts(slices.Concat(s...), grid.Height*grid.Width*4)
}

// windowIndex teilt das Grid in Fenster auf und gibt zurueck:
//  1. Slice von Grid-Punkt-Indizes organisiert nach Fenstern
//  2. Slice von Grenzen fuer Start/Ende jedes Fensters
func (m *VisionModel) windowIndex(grid *Grid) (index []int32, bounds []int) {
	height := grid.Height / m.spatialMergeSize
	width := grid.Width / m.spatialMergeSize
	window := m.windowSize / m.patchSize / m.spatialMergeSize

	index = make([]int32, height*width)

	// Bounds-Array vorallokieren
	numWindows := ((height + window - 1) / window) * ((width + window - 1) / window)
	bounds = make([]int, 0, numWindows+1)
	bounds = append(bounds, 0)

	var cur int32
	for y := 0; y < height; y += window {
		for x := 0; x < width; x += window {
			h1 := min(window, height-y)
			w1 := min(window, width-x)
			for dy := range h1 {
				for dx := range w1 {
					win := (y+dy)*width + (x + dx)
					index[win] = cur
					cur++
				}
			}
			bounds = append(bounds, int(cur)*window)
		}
	}
	return index, bounds
}

// blockDiagonalMask erstellt eine Block-Diagonale Attention-Maske
func blockDiagonalMask(ctx ml.Context, seqLength int, bounds []int) ml.Tensor {
	// 2D-Maske mit -Inf initialisieren (kein Attention erlaubt)
	s := make([][]float32, seqLength)
	negInf := float32(math.Inf(-1))
	for i := range s {
		s[i] = slices.Repeat([]float32{negInf}, seqLength)
	}

	// Maske mit Nullen fuer erlaubte Attention innerhalb von Bloecken fuellen
	for i := 1; i < len(bounds); i++ {
		start, end := bounds[i-1], bounds[i]
		// Attention innerhalb dieses Sequence-Blocks aktivieren
		for row := start; row < end; row++ {
			for col := start; col < end; col++ {
				s[row][col] = 0.0
			}
		}
	}

	return ctx.Input().FromFloats(slices.Concat(s...), seqLength, seqLength)
}

// newVisionModel erstellt eine neue Instanz des Qwen Vision-Modells
func newVisionModel(c fs.Config) *VisionModel {
	patchSize := int(c.Uint("vision.patch_size", 14))
	hiddenSize := int(c.Uint("vision.embedding_length", 1280))
	numHeads := int(c.Uint("vision.attention.head_count", 16))
	numChannels := int(c.Uint("vision.num_channels", 3))
	eps := c.Float("vision.attention.layer_norm_epsilon", 1e-6)
	ropeTheta := c.Float("vision.rope.freq_base", 10000.0)
	spatialMergeSize := int(c.Uint("vision.spatial_merge_size", 2))
	windowSize := int(c.Uint("vision.window_size", 112))
	fullAttnBlocks := c.Ints("qwen25vl.vision.fullatt_block_indexes", []int32{7, 15, 23, 31})
	temporalPatchSize := int(c.Uint("vision.temporal_patch_size", 2))

	model := &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 32)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:        hiddenSize,
			numHeads:          numHeads,
			headDim:           hiddenSize / numHeads,
			patchSize:         patchSize,
			numChannels:       numChannels,
			eps:               eps,
			ropeTheta:         ropeTheta,
			spatialMergeSize:  spatialMergeSize,
			windowSize:        windowSize,
			temporalPatchSize: temporalPatchSize,
			fullAttnBlocks:    fullAttnBlocks,
		},
	}

	return model
}
