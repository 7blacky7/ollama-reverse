// Package lfm2 - Slot-Management fuer HybridCache
// Dieses Modul enthaelt die Slot-Allokation, Copy-on-Write-Logik
// und Sequence-Verwaltung (CopyPrefix, Remove, EnsureWritable).
package lfm2

import (
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

func (c *HybridCache) allocSlot() (int, error) {
	if len(c.freeSlots) == 0 {
		return 0, kvcache.ErrKvCacheFull
	}
	slot := c.freeSlots[len(c.freeSlots)-1]
	c.freeSlots = c.freeSlots[:len(c.freeSlots)-1]
	return slot, nil
}

func (c *HybridCache) freeSlot(slot int) {
	// Bounds check before freeing
	if slot >= 0 && slot < c.maxSequences {
		c.freeSlots = append(c.freeSlots, slot)
	}
}

// zeroConvSlots zeros the conv state for the given slots across all layers.
// This must be called when recycling slots to prevent stale state from affecting new sequences.
func (c *HybridCache) zeroConvSlots(ctx ml.Context, slots []int) {
	if len(slots) == 0 || len(c.convStates) == 0 {
		return
	}

	// Use input context for creating tensors
	inputCtx := ctx.Input()

	// Create slot indices tensor
	slotIndices := make([]int32, len(slots))
	for i, s := range slots {
		slotIndices[i] = int32(s)
	}
	slotsTensor := inputCtx.FromInts(slotIndices, len(slotIndices))

	// Create zero tensor for the slots (SetRows requires F32 source)
	zeros := inputCtx.Zeros(ml.DTypeF32, c.dConv*c.hiddenSize, len(slots))

	// Zero each layer's conv state for these slots
	for _, buf := range c.convStates {
		ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
	}
}

// EnsureWritable ensures that sequences in the current batch have private (non-shared) conv slots.
// Returns an error if slot allocation fails.
func (c *HybridCache) EnsureWritable(ctx ml.Context) error {
	for i, seq := range c.curSeqs {
		slot, ok := c.slotForSeq[seq]
		if !ok {
			continue
		}

		// Bounds check
		if slot < 0 || slot >= len(c.refCount) {
			continue
		}

		if c.refCount[slot] <= 1 {
			continue
		}

		newSlot, err := c.allocSlot()
		if err != nil {
			return err
		}
		c.refCount[slot]--
		c.refCount[newSlot] = 1
		c.slotForSeq[seq] = newSlot
		c.curSlots[i] = newSlot

		// Copy existing conv state for all initialized layers
		for _, buf := range c.convStates {
			// buf: [dConv*hiddenSize, maxSlots]
			src := buf.Rows(ctx, ctx.Input().FromInts([]int32{int32(slot)}, 1))
			// SetRows requires F32 source
			srcF32 := src.Cast(ctx, ml.DTypeF32)
			ctx.Forward(buf.SetRows(ctx, srcF32, ctx.Input().FromInts([]int32{int32(newSlot)}, 1)))
		}
	}

	// Rebuild current slots tensor
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	return nil
}

func (c *HybridCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	// KV cache shares prefix metadata (no copy) which is correct for prefix reuse.
	c.kv.CopyPrefix(srcSeq, dstSeq, prefixLen)

	// For shortconv state we implement copy-on-write: dst shares the same slot as src.
	// On the first write to dst, EnsureWritable will create a private slot.
	if dstSlot, ok := c.slotForSeq[dstSeq]; ok {
		// Bounds check before decrementing
		if dstSlot >= 0 && dstSlot < len(c.refCount) {
			c.refCount[dstSlot]--
			if c.refCount[dstSlot] <= 0 {
				c.refCount[dstSlot] = 0
				c.freeSlot(dstSlot)
			}
		}
		delete(c.slotForSeq, dstSeq)
	}

	srcSlot, ok := c.slotForSeq[srcSeq]
	if !ok {
		// src may not have a slot yet; dst will allocate on demand
		return
	}

	// Bounds check before incrementing
	if srcSlot >= 0 && srcSlot < len(c.refCount) {
		c.slotForSeq[dstSeq] = srcSlot
		c.refCount[srcSlot]++
	}
}

func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if err := c.kv.Remove(seq, beginIndex, endIndex); err != nil {
		return err
	}

	// For recurrent state, any removal invalidates the state because
	// the state at position N depends on all previous positions.
	// Drop the slot mapping so it resets on next use.
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return nil
	}

	// Bounds check
	if slot < 0 || slot >= len(c.refCount) {
		delete(c.slotForSeq, seq)
		return nil
	}

	c.refCount[slot]--
	if c.refCount[slot] <= 0 {
		c.refCount[slot] = 0
		c.freeSlot(slot)
	}
	delete(c.slotForSeq, seq)

	return nil
}

// Seqs returns the ordered unique sequences for the current forward pass.
func (c *HybridCache) Seqs() []int {
	return slices.Clone(c.curSeqs)
}
