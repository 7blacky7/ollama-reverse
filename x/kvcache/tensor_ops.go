// Package kvcache - Tensor-Operationen (Legacy/Kommentiert)
//
// Dieses Modul enthaelt die auskommentierten Get/Put-Operationen
// mit alternativer Tensor-Manipulation via TakeAxes/Scatter
// statt View/SetRows. Enthaelt umfangreiches Debug-Logging.
// Unterschied zur Produktionsversion: Speichert Head-Dimensionen
// pro Layer in separaten Maps.
package kvcache

// import (
// 	"fmt"
// 	"log/slog"
// 	"slices"

// 	"github.com/ollama/ollama/ml"
// )

// func (c *Causal) SetLayer(layer int) {
// 	c.curLayer = layer
// }

// // SetCausal disables causal mask generation for a particular range of indicies in
// // the current batch for subsequent calls to Get. The state resets for the next forward pass.
// func (c *Causal) SetCausal(ctx ml.Context, opts CausalOptions) {
// 	if !slices.Equal(c.opts.Except, opts.Except) {
// 		c.opts = opts
// 		if ctx != nil {
// 			c.curMask = c.buildMask(ctx)
// 		}
// 	}
// }

// func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
// 	key := c.keys[c.curLayer]
// 	value := c.values[c.curLayer]

// 	kHeadDim := c.kHeadDims[c.curLayer]
// 	vHeadDim := c.vHeadDims[c.curLayer]
// 	numKVHeads := c.numKVHeads[c.curLayer]
// 	// rowSize := numKVHeads * c.curBatchSize
// 	// cachedSize := c.curMask.Dim(1)
// 	cachedSize := c.curLoc.Dim(0)
// 	// kCellSize := kHeadDim * numKVHeads
// 	// vCellSize := vHeadDim * numKVHeads

// 	slog.Info("XXX Causal.Get full cache", "key", key)
// 	slog.Info("XXX Causal.Get full cache", "value", value)
// 	slog.Info("XXX Causal.Get full cache", "curloc", c.curLoc)
// 	slog.Info("XXX Causal.Get", "curMask", c.curMask)
// 	slog.Info("XXX Causal.Get", "kHeadDim", kHeadDim, "numKVHeads", numKVHeads, "cachedSize", cachedSize, "kHeadDim", kHeadDim)
// 	// panic("XXX")

// 	// fmt.Fprintln(os.Stderr, key.ToString())
// 	// panic("full cache value")

// 	// TODO we should use TakeAxes to gather the cells from curLoc, but for now to be consistent with GGML, just grab a larger chunk and mask
// 	key = key.TakeAxes(ctx, c.curLoc, 0).Reshape(ctx, 1, numKVHeads, cachedSize, kHeadDim)
// 	// key = key.AsStrided(ctx, []int{1, numKVHeads, cachedSize, kHeadDim}, []int{}, rowSize*c.curCellRange.min)

// 	// slog.Info("XXX Causal.Get after AsStrided", "key", key)
// 	// panic("XXX")

// 	// if c.config.PermutedV {
// 	// 	panic("permuted")
// 	// 	// TODO not converted
// 	// 	vHeadDim := value.Dim(1)
// 	// 	elemSize := value.Stride(2)

// 	// 	value = value.AsStrided(ctx,
// 	// 		[]int{numKVHeads, vHeadDim, cachedSize},
// 	// 		[]int{value.Stride(0), value.Stride(1)},
// 	// 		elemSize*c.curCellRange.min,
// 	// 	)
// 	// } else {
// 	// vHeadDim := c.vHeadDims[c.curLayer]
// 	// rowSize := value.Stride(2)
// 	// slog.Info("XXX Causal.Get before AsStrided", "vHeadDim", vHeadDim, "rowSize", rowSize)
// 	// panic("XXX")

// 	// TODO we should use TakeAxes to gather the cells from curLoc, but for now to be consistent with GGML, just grab a larger chunk and mask
// 	value = value.TakeAxes(ctx, c.curLoc, 0).Reshape(ctx, 1, numKVHeads, cachedSize, vHeadDim)
// 	// value = value.AsStrided(ctx, []int{1, numKVHeads, cachedSize, vHeadDim}, []int{}, rowSize*c.curCellRange.min)

// 	// slog.Info("XXX Causal.Get after AsStrided", "value", value)
// 	// panic("XXX")

// 	// }

// 	// // TODO The mask changes from X,X to 1,X, and with the Row-order change
// 	// // the 1 becomes trailing and messes up later operations
// 	// // This isn't the right solution, but works around it...
// 	// if c.curMask.Dim(1) == 1 {
// 	// 	return key, value, c.curMask.Transpose(ctx, 1, 0, 2, 3)
// 	// }
// 	// fmt.Fprintln(os.Stderr, key.ToString())
// 	// fmt.Fprintln(os.Stderr, value.ToString())
// 	// panic("XXX")
// 	slog.Info("XXX Mask", "curLayer", c.curLayer, "shape", c.curMask.Shape())

// 	return key, value, c.curMask
// }

// func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
// 	kHeadDim := key.Dim(3)
// 	vHeadDim := value.Dim(3)
// 	numKVHeads := key.Dim(1)
// 	batchSize := key.Dim(2)
// 	kCellSize := kHeadDim * numKVHeads
// 	vCellSize := vHeadDim * numKVHeads

// 	// slog.Info("XXX Causal.Put", "key", key, "value", value)
// 	slog.Info("XXX Causal.Put", "kHeadDim", kHeadDim, "vHeadDim", vHeadDim, "numKVHeads", numKVHeads, "batchSize", batchSize)
// 	// panic("XXX")

// 	if c.curBatchSize != batchSize {
// 		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, batchSize))
// 	}

// 	// slog.Info("XXX", "c.ctxs", c.ctxs, "c.curLayer", c.curLayer, "backend", c.backend)
// 	if _, ok := c.ctxs[c.curLayer]; !ok {
// 		slog.Info("XXX Causal.Put creating new context", "c.curLayer", c.curLayer)
// 		c.ctxs[c.curLayer] = c.backend.NewContext().Layer(c.curLayer)
// 	}

// 	if _, ok := c.keys[c.curLayer]; !ok {
// 		slog.Info("XXX Causal.Put allocating keys", "c.curLayer", c.curLayer, "shape", []int{len(c.cells), kCellSize})

// 		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, len(c.cells), kCellSize)
// 		c.kHeadDims[c.curLayer] = kHeadDim
// 		c.vHeadDims[c.curLayer] = vHeadDim
// 		c.numKVHeads[c.curLayer] = numKVHeads
// 	}

// 	if _, ok := c.values[c.curLayer]; !ok {
// 		// if c.config.PermutedV {
// 		// 	c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, numKVHeads, vHeadDim, len(c.cells))
// 		// } else {
// 		c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, len(c.cells), vCellSize)
// 		// }
// 	}

// 	key = key.Reshape(ctx, batchSize, 1, kCellSize) //.Contiguous(ctx, false) // TODO contiguous may not be needed

// 	// slog.Info("XXX Causal.Put after reshape", "keyCache", keyCache)
// 	// panic("XXX")
// 	// curLoc := 0 // TODO c.curLoc is now a tensor
// 	// kSize := numKVHeads * kHeadDim
// 	// vSize := numKVHeads * vHeadDim
// 	// start := []int{int(curLoc), 0}
// 	// kStop := []int{int(curLoc + batchSize), int(kSize)}
// 	// vStop := []int{int(curLoc + batchSize), int(vSize)}
// 	// strides := []int{1, 1}

// 	// slog.Info("XXX Causal.Put Key SliceUpdate", "keyCache", keyCache)
// 	// slog.Info("XXX Causal.Put Key SliceUpdate", "key", key)

// 	// slog.Info("XXX Causal.Put Key SliceUpdate", "start", start, "kStop", kStop, "strides", strides)

// 	// ctx.Forward(c.keys[c.curLayer].SliceUpdate(ctx, key, start, kStop, strides))
// 	ctx.Forward(c.keys[c.curLayer].Scatter(ctx, []ml.Tensor{c.curLoc}, key, []int{0}))
// 	// fmt.Fprintln(os.Stderr, keyCache.ToString())
// 	// panic("input value")

// 	// fmt.Fprintln(os.Stderr, t.ToString())
// 	// panic("XXX")

// 	// if c.config.PermutedV {
// 	// 	panic("permuted")
// 	// 	// TODO not adjusted
// 	// 	value = value.Reshape(ctx, vHeadDim*numKVHeads, 1, batchSize)
// 	// 	value = value.Transpose(ctx, 2, 0, 1, 3)

// 	// 	valueCache := c.values[c.curLayer]
// 	// 	valueCache = valueCache.Reshape(ctx, 1, len(c.cells), vHeadDim*numKVHeads)

// 	// 	ctx.Forward(valueCache.SliceUpdate(ctx, value, start, vStop, strides))
// 	// } else {
// 	value = value.Reshape(ctx, batchSize, 1, vCellSize) //.Contiguous(ctx, false) // TODO contiguous may not be needed
// 	// slog.Info("XXX Causal.Put Value SliceUpdate", "valueCache", valueCache)
// 	// slog.Info("XXX Causal.Put Value SliceUpdate", "value", value)
// 	// slog.Info("XXX Causal.Put Value SliceUpdate", "start", start, "vStop", vStop, "strides", strides)

// 	ctx.Forward(c.values[c.curLayer].Scatter(ctx, []ml.Tensor{c.curLoc}, value, []int{0}))
// 	// }
// 	// fmt.Fprintln(os.Stderr, c.keys[c.curLayer].ToString())
// 	// fmt.Fprintln(os.Stderr, c.values[c.curLayer].ToString())
// 	// panic("XXX")

// }
