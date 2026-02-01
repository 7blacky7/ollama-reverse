package qwen25vl

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// ============================================================================
// Vision Patch - Patch-Embedding und Patch-Merging fuer das Vision-Modell
// ============================================================================
//
// Dieses Modul enthaelt:
// - PatchEmbedding: Konvertiert Bildpatches zu Embeddings via 2D-Convolution
// - VisionPatchMerger: Merged Patches fuer effizientere Verarbeitung

// Convolution-Parameter (konstant fuer Patch-Embedding)
const (
	patchConvPadding  = 0 // Kein Padding
	patchConvDilation = 1 // Standard-Dilation
)

// PatchEmbedding konvertiert Bildpatches zu Embeddings
type PatchEmbedding struct {
	PatchConv0 *nn.Conv2D `gguf:"patch_embd_0"`
	PatchConv1 *nn.Conv2D `gguf:"patch_embd_1"`
}

// Forward verarbeitet Pixel-Werte zu Patch-Embeddings
func (pe *PatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	numPatches := pixelValues.Shape()[1]

	// Eingabe-Tensor umformen fuer erwartete Dimensionen
	pixelValues = pixelValues.Reshape(ctx, opts.patchSize*opts.patchSize, opts.temporalPatchSize, opts.numChannels, numPatches)

	// Tensor permutieren fuer temporale Dimension
	pixelValues = pixelValues.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Tensor aufteilen fuer temporale Convolutions
	in0 := pe.extractTemporalSlice(ctx, pixelValues, 0, opts, numPatches)
	in1 := pe.extractTemporalSlice(ctx, pixelValues, 1, opts, numPatches)

	// Convolution mit vollem Stride
	stride := opts.patchSize
	out0 := pe.PatchConv0.Forward(ctx, in0, stride, stride, patchConvPadding, patchConvPadding, patchConvDilation, patchConvDilation)
	out1 := pe.PatchConv1.Forward(ctx, in1, stride, stride, patchConvPadding, patchConvPadding, patchConvDilation, patchConvDilation)

	// Ausgaben der temporalen Convolutions addieren
	out := out0.Add(ctx, out1)

	// Ausgabe-Tensor umformen
	return out.Reshape(ctx, opts.hiddenSize, numPatches)
}

// extractTemporalSlice extrahiert einen temporalen Slice aus dem Tensor
func (pe *PatchEmbedding) extractTemporalSlice(ctx ml.Context, pixelValues ml.Tensor, temporalIndex int, opts *VisionModelOptions, numPatches int) ml.Tensor {
	offset := temporalIndex * pixelValues.Stride(0)
	slice := pixelValues.View(ctx,
		offset, 1,
		pixelValues.Stride(1), pixelValues.Dim(1),
		pixelValues.Stride(2), pixelValues.Dim(2),
		pixelValues.Stride(3), pixelValues.Dim(3),
	).Contiguous(ctx)
	return slice.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)
}

// VisionPatchMerger merged Patches fuer effizientere Verarbeitung
type VisionPatchMerger struct {
	LNQ  *nn.RMSNorm `gguf:"ln_q"`
	MLP0 *nn.Linear  `gguf:"mlp.0"`
	MLP2 *nn.Linear  `gguf:"mlp.2"`
}

// Forward fuehrt Patch-Merging durch
func (pm *VisionPatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	normalized := pm.LNQ.Forward(ctx, visionOutputs, opts.eps)

	// Hidden-Size berechnen basierend auf Spatial-Merge-Groesse
	spatialMergeArea := opts.spatialMergeSize * opts.spatialMergeSize
	hiddenSize := visionOutputs.Dim(0) * spatialMergeArea

	// Normalisierte Ausgabe umformen fuer MLP
	reshaped := normalized.Reshape(ctx, hiddenSize, normalized.Dim(1)/spatialMergeArea)
	hidden := pm.MLP0.Forward(ctx, reshaped)
	activated := hidden.GELU(ctx)

	return pm.MLP2.Forward(ctx, activated)
}
