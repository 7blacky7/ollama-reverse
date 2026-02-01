// Package kvcache - Masken-Generierung (Legacy/Kommentiert)
//
// Dieses Modul enthaelt die auskommentierte buildMask-Funktion
// mit erweitertem Batch-Padding und Debug-Logging.
// Unterschied zur Produktionsversion: Verwendet MaskBatchPadding
// fuer die Batch-Dimension-Ausrichtung.
package kvcache

// import (
// 	"log/slog"
// 	"math"
// 	"slices"

// 	"github.com/ollama/ollama/ml"
// )

// // Builds a mask of history x batch indicating whether for each token in the batch the
// // token in the history should apply. This is based on both the sequence and causality (the
// // position of the history is not ahead of the token in the batch).
// func (c *Causal) buildMask(ctx ml.Context) ml.Tensor {
// 	// Align and pad the two dimensions as required by the backend
// 	batchSize := roundUp(c.curBatchSize, c.config.MaskBatchPadding)

// 	c.curCellRange.min = roundDown(c.curCellRange.min, c.config.CachePadding)
// 	c.curCellRange.max = roundUp(c.curCellRange.max+1, c.config.CachePadding) - 1

// 	length := c.curCellRange.max - c.curCellRange.min + 1

// 	mask := make([]float32, batchSize*length)

// 	for i := range c.curBatchSize {
// 		enabled := !slices.Contains(c.opts.Except, i)
// 		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
// 			if !slices.Contains(c.cells[j].sequences, c.curSequences[i]) ||
// 				(enabled && c.cells[j].pos > c.curPositions[i]) ||
// 				c.chunkSize > 0 && c.cells[j].pos < c.curPositions[i]-c.curPositions[i]%c.chunkSize ||
// 				c.cells[j].pos < c.curPositions[i]-c.swaWindowSize {
// 				mask[i*length+(j-c.curCellRange.min)] = float32(math.Inf(-1))
// 			}
// 		}
// 	}

// 	// Mask out any padding tokens we added. For padding that we added to the cache history, this
// 	// has already been masked out because the sequence doesn't match.
// 	for i := c.curBatchSize * length; i < len(mask); i++ {
// 		mask[i] = float32(math.Inf(-1))
// 	}

// 	maskTensor := ctx.Input().FromFloats(mask, batchSize, length)

// 	// if c.config.MaskDType != ml.DTypeFloat32 {
// 	// 	maskTensor = maskTensor.Cast(ctx, c.config.MaskDType)
// 	// }

// 	slog.Info("XXX Causal.buildMask", "c.curBatchSize", c.curBatchSize, "c.config.MaskBatchPadding", c.config.MaskBatchPadding, "c.curCellRange.min", c.curCellRange.min, "c.curCellRange.max", c.curCellRange.max, "size", len(mask), "shape", []int{1, batchSize, length})

// 	return maskTensor
// }
