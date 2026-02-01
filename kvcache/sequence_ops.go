// Package kvcache - Sequenz-Operationen
//
// Dieses Modul verwaltet Sequenz-bezogene Operationen:
// - CopyPrefix: Kopiert einen Praefix von einer Sequenz zu einer anderen
// - CanResume: Prueft ob eine Sequenz fortgesetzt werden kann
// - shift: Verschiebt Positionen im Cache (RoPE-Anpassung)
// - Remove: Entfernt Tokens aus einer Sequenz
package kvcache

import (
	"errors"
	"math"
	"slices"
)

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, len int32) {
	seqRange := newRange()

	for i := range c.cells {
		// Remove the contents of dstSeq so that we only have the copied prefix, metadata will be reset at the end
		if slices.Contains(c.cells[i].sequences, dstSeq) {
			c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == dstSeq })
		}

		if slices.Contains(c.cells[i].sequences, srcSeq) && c.cells[i].pos < len {
			c.cells[i].sequences = append(c.cells[i].sequences, dstSeq)
			if i < seqRange.min {
				seqRange.min = i
			}
			if i > seqRange.max {
				seqRange.max = i
			}
		}
	}

	c.cellRanges[dstSeq] = seqRange
}

func (c *Causal) CanResume(seq int, pos int32) bool {
	if c.swaMemorySize == math.MaxInt32 {
		return true
	}

	seqRange, ok := c.cellRanges[seq]
	if !ok {
		return false
	}

	// for sliding window, check that the window of the new sequence is contained in
	// the window of what we are storing
	var first int32 = math.MaxInt32
	var last int32 = -1
	for i := seqRange.min; i <= seqRange.max; i++ {
		if slices.Contains(c.cells[i].sequences, seq) {
			first = min(first, c.cells[i].pos)
			last = max(last, c.cells[i].pos)
		}
	}

	if last == -1 {
		return false
	}

	posWindowStart := max(0, pos-c.swaWindowSize)
	return posWindowStart >= first && pos <= last+1
}

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	if c.shiftFn == nil {
		return ErrNotSupported
	}

	seqRange := c.cellRanges[seq]

	for start := seqRange.min; start <= seqRange.max; start += c.maxBatch {
		size := min(seqRange.max-start+1, c.maxBatch)
		offsets := make([]int32, size)

		var batchFirst, batchLast int

		batchFirst = -1
		for i := range offsets {
			cell := c.cells[start+i]

			if slices.Contains(cell.sequences, seq) && cell.pos >= beginIndex {
				offsets[i] = offset
				if batchFirst < 0 {
					batchFirst = i
				}
				batchLast = i
			}
		}

		if batchFirst < 0 {
			continue
		}

		offsets = offsets[batchFirst : batchLast+1]

		ctx := c.backend.NewContext()
		kShift := ctx.Input().FromInts(offsets, len(offsets))

		for i, key := range c.keys {
			if key == nil {
				continue
			}

			kHeadDim := key.Dim(0)
			numKVHeads := key.Dim(1)
			rowSize := key.Stride(2)

			key = key.View(ctx, rowSize*(start+batchFirst),
				kHeadDim, key.Stride(1),
				numKVHeads, key.Stride(2),
				len(offsets),
			)

			roped, err := c.shiftFn(ctx, i, key, kShift)
			if err != nil {
				ctx.Close()
				return err
			}

			ctx.Forward(roped.Copy(ctx, key))
		}

		ctx.Compute()
		ctx.Close()
	}

	return nil
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
	// TODO(jessegross): We should check to see if removing the middle of the sequence will
	// cause the sliding window to encompass tokens that we no longer have. If so, then we
	// should return an error, which will trigger the runner to evaluate the full history and
	// rebuild the window. However, if we have multimodal inputs in our history, this reuse
	// results in use after free, so we don't do it for now.

	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range c.cells {
		if slices.Contains(c.cells[i].sequences, seq) {
			if c.cells[i].pos >= beginIndex && c.cells[i].pos < endIndex {
				c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == seq })
			} else {
				if c.cells[i].pos >= endIndex {
					if slices.ContainsFunc(c.cells[i].sequences, func(s int) bool { return s != seq }) {
						return errors.New("shifting cells shared by multiple sequences not supported")
					}

					c.cells[i].pos += offset
				}
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}
	}

	if seqRange == newRange() {
		delete(c.cellRanges, seq)
		return nil
	}

	c.cellRanges[seq] = seqRange

	if endIndex != math.MaxInt32 {
		err := c.shift(seq, endIndex+offset, offset)
		if err != nil {
			return err
		}
	}

	return nil
}
