// Package lfm2 - Conv-State-Management fuer HybridCache
// Dieses Modul enthaelt die Logik fuer ConvState-Lesen und -Schreiben
// sowie den Conv-Buffer-Manager fuer Layer-spezifische Zustaende.
package lfm2

import (
	"github.com/ollama/ollama/ml"
)

func (c *HybridCache) convBuffer(ctx ml.Context, layer int) ml.Tensor {
	if buf, ok := c.convStates[layer]; ok {
		return buf
	}

	if _, ok := c.convCtxs[layer]; !ok {
		c.convCtxs[layer] = c.backend.NewContextSize(1).Layer(layer)
	}

	buf := c.convCtxs[layer].Zeros(c.dtype, c.dConv*c.hiddenSize, c.maxSequences)
	c.convStates[layer] = buf
	return buf
}

// ConvState returns the conv state for current batch sequences as shape [dConv, hiddenSize, nSeqs].
// Returns an error if copy-on-write allocation fails.
func (c *HybridCache) ConvState(ctx ml.Context, layer int) (ml.Tensor, error) {
	if !c.writableEnsured {
		needsWritable := false
		for _, seq := range c.curSeqs {
			slot, ok := c.slotForSeq[seq]
			if !ok {
				continue
			}
			if slot >= 0 && slot < len(c.refCount) && c.refCount[slot] > 1 {
				needsWritable = true
				break
			}
		}

		if needsWritable {
			if err := c.EnsureWritable(ctx); err != nil {
				c.writableError = err
			}
		}
		c.writableEnsured = true
	}

	if c.writableError != nil {
		return nil, c.writableError
	}

	buf := c.convBuffer(ctx, layer)
	cur := buf.Rows(ctx, c.slotsTensor())
	return cur.Reshape(ctx, c.dConv, c.hiddenSize, c.numSeqs()), nil
}

// UpdateConvState writes a new conv state for current batch sequences.
// newState must have shape [dConv, hiddenSize, nSeqs].
func (c *HybridCache) UpdateConvState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.convBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.dConv*c.hiddenSize, c.numSeqs())
	// SetRows requires F32 source
	srcF32 := src.Cast(ctx, ml.DTypeF32)
	ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
}

// IsSupportedForBatch returns true if the current batch layout supports shortconv.
func (c *HybridCache) IsSupportedForBatch() bool {
	return c.curSeqTokens > 0 && len(c.curSeqs) > 0
}
