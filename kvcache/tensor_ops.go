// Package kvcache - Tensor-Operationen (Get/Put)
//
// Dieses Modul enthaelt die Kern-Tensor-Operationen:
// - SetLayer: Setzt den aktiven Layer
// - SetCausal: Konfiguriert die Kausalitaetsmaske
// - Get: Liest Key/Value-Tensoren aus dem Cache
// - Put: Schreibt Key/Value-Tensoren in den Cache
package kvcache

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/ml"
)

func (c *Causal) SetLayer(layer int) {
	c.curLayer = layer
}

// SetCausal disables causal mask generation for a particular range of indicies in
// the current batch for subsequent calls to Get. The state resets for the next forward pass.
func (c *Causal) SetCausal(ctx ml.Context, opts CausalOptions) {
	if !slices.Equal(c.opts.Except, opts.Except) {
		c.opts = opts
		if ctx != nil {
			c.curMask = c.buildMask(ctx)
		}
	}
}

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	kHeadDim := key.Dim(0)
	numKVHeads := key.Dim(1)
	rowSize := key.Stride(2)
	cachedSize := c.curMask.Dim(0)

	key = key.View(ctx, rowSize*c.curCellRange.min,
		kHeadDim, key.Stride(1),
		numKVHeads, key.Stride(2),
		cachedSize,
	)

	if c.config.PermutedV {
		vHeadDim := value.Dim(1)
		elemSize := value.Stride(0)

		value = value.View(ctx, elemSize*c.curCellRange.min,
			cachedSize, value.Stride(1),
			vHeadDim, value.Stride(2),
			numKVHeads,
		)
	} else {
		vHeadDim := value.Dim(0)
		rowSize := value.Stride(2)

		value = value.View(ctx, rowSize*c.curCellRange.min,
			vHeadDim, value.Stride(1),
			numKVHeads, value.Stride(2),
			cachedSize,
		)
	}

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(0)
	vHeadDim := value.Dim(0)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)

	if c.curBatchSize != batchSize {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, batchSize))
	}

	if _, ok := c.ctxs[c.curLayer]; !ok {
		c.ctxs[c.curLayer] = c.backend.NewContextSize(2).Layer(c.curLayer)
	}

	if _, ok := c.keys[c.curLayer]; !ok {
		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, kHeadDim, numKVHeads, len(c.cells))
	}

	if _, ok := c.values[c.curLayer]; !ok {
		if c.config.PermutedV {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, len(c.cells), vHeadDim, numKVHeads)
		} else {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, vHeadDim, numKVHeads, len(c.cells))
		}
	}

	key = key.Reshape(ctx, kHeadDim*numKVHeads, batchSize)
	keyCache := c.keys[c.curLayer]
	keyCache = keyCache.Reshape(ctx, kHeadDim*numKVHeads, len(c.cells))
	ctx.Forward(keyCache.SetRows(ctx, key, c.curLoc))

	if c.config.PermutedV {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, 1, batchSize)
		value = value.Permute(ctx, 2, 0, 1, 3)

		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, 1, len(c.cells), vHeadDim*numKVHeads)

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	} else {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, batchSize)
		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, vHeadDim*numKVHeads, len(c.cells))

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	}
}
