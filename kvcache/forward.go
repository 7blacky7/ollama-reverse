// Package kvcache - Forward Pass Operationen
//
// Dieses Modul enthaelt die Kernlogik fuer den Forward Pass:
// - StartForward: Initialisiert einen neuen Forward Pass
// - findLocs: Findet freie Cache-Positionen
// - updateSlidingWindow: Verwaltet das Sliding Window
// - buildMask: Erstellt die Attention-Maske
// - Hilfsfunktionen: roundUp, roundDown, newRange
package kvcache

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func (c *Causal) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.curBatchSize = len(batch.Positions)
	c.curSequences = batch.Sequences
	c.curPositions = batch.Positions
	c.opts.Except = nil

	var locs []int32
	if !reserve {
		c.updateSlidingWindow()

		var err error
		locs, err = c.findLocs()
		if err != nil {
			return err
		}

		for i, pos := range batch.Positions {
			seq := batch.Sequences[i]
			loc := int(locs[i])

			c.cells[loc] = cacheCell{pos: pos, sequences: []int{seq}}

			seqRange, ok := c.cellRanges[seq]
			if !ok {
				seqRange = newRange()
			}

			seqRange.min = min(seqRange.min, loc)
			c.curCellRange.min = min(c.curCellRange.min, loc)

			seqRange.max = max(seqRange.max, loc)
			c.curCellRange.max = max(c.curCellRange.max, loc)

			c.cellRanges[seq] = seqRange
		}
	} else {
		// If we are reserving memory, don't update any of the cache metadata but set the size
		// to the worst case.
		locs = make([]int32, c.curBatchSize)
		for i := range locs {
			locs[i] = int32(i)
		}
		c.curCellRange.min = 0
		c.curCellRange.max = len(c.cells) - 1
	}

	c.curLoc = ctx.Input().FromInts(locs, len(locs))
	c.curMask = c.buildMask(ctx)

	return nil
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

// Returns a slice of locations where each token in the batch should be stored
func (c *Causal) findLocs() ([]int32, error) {
	loc := make([]int32, 0, c.curBatchSize)

	for i := range c.cells {
		if len(c.cells[i].sequences) == 0 {
			loc = append(loc, int32(i))
			if len(loc) >= c.curBatchSize {
				return loc, nil
			}
		}
	}

	return nil, fmt.Errorf("%w (cache: %v batch: %v)", ErrKvCacheFull, len(c.cells), c.curBatchSize)
}

func (c *Causal) updateSlidingWindow() {
	c.curCellRange = newRange()

	if c.swaMemorySize == math.MaxInt32 {
		for _, seq := range c.curSequences {
			if seqRange, ok := c.cellRanges[seq]; ok {
				c.curCellRange.min = min(c.curCellRange.min, seqRange.min)
				c.curCellRange.max = max(c.curCellRange.max, seqRange.max)
			}
		}

		return
	}

	type lowestPosition struct {
		pos      int32
		curBatch bool
	}

	// create a map of unique sequences to the lowest position in that sequence
	lowestPos := make(map[int]lowestPosition)
	for i := range c.curPositions {
		seq := c.curSequences[i]

		lowest, ok := lowestPos[seq]
		if !ok {
			lowest = lowestPosition{pos: c.curPositions[i], curBatch: true}
		} else if c.curPositions[i] < lowest.pos {
			lowest.pos = c.curPositions[i]
		}

		lowestPos[seq] = lowest
	}

	// for any sequences are not part of this batch, clean up any tokens
	// that are no longer needed after the processing of the previous
	// batch
	for seq, seqRange := range c.cellRanges {
		if _, ok := lowestPos[seq]; !ok {
			var last int32
			for i := seqRange.min; i <= seqRange.max; i++ {
				if slices.Contains(c.cells[i].sequences, seq) {
					last = max(last, c.cells[i].pos)
				}
			}

			lowestPos[seq] = lowestPosition{pos: last + 1, curBatch: false}
		}
	}

	// delete any entries that are beyond the window of the oldest position in the sequence
	for seq, lowest := range lowestPos {
		oldRange, ok := c.cellRanges[seq]
		if !ok {
			continue
		}

		newRange := newRange()

		for i := oldRange.min; i <= oldRange.max; i++ {
			if slices.Contains(c.cells[i].sequences, seq) {
				if c.cells[i].pos < lowest.pos-c.swaMemorySize {
					c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == seq })
				} else {
					newRange.min = min(newRange.min, i)
					newRange.max = max(newRange.max, i)
				}
				if lowest.curBatch && c.cells[i].pos >= lowest.pos-c.swaWindowSize {
					c.curCellRange.min = min(c.curCellRange.min, i)
					c.curCellRange.max = max(c.curCellRange.max, i)
				}
			}
		}

		c.cellRanges[seq] = newRange
	}
}

func roundDown(length, pad int) int {
	return (length / pad) * pad
}

func roundUp(length, pad int) int {
	return ((length + pad - 1) / pad) * pad
}

// Builds a mask of history x batch indicating whether for each token in the batch the
// token in the history should apply. This is based on both the sequence and causality (the
// position of the history is not ahead of the token in the batch).
func (c *Causal) buildMask(ctx ml.Context) ml.Tensor {
	c.curCellRange.min = roundDown(c.curCellRange.min, c.config.CachePadding)
	c.curCellRange.max = roundUp(c.curCellRange.max+1, c.config.CachePadding) - 1

	length := c.curCellRange.max - c.curCellRange.min + 1

	mask := make([]float32, c.curBatchSize*length)

	for i := range c.curBatchSize {
		enabled := !slices.Contains(c.opts.Except, i)
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			if !slices.Contains(c.cells[j].sequences, c.curSequences[i]) ||
				(enabled && c.cells[j].pos > c.curPositions[i]) ||
				c.chunkSize > 0 && c.cells[j].pos < c.curPositions[i]-c.curPositions[i]%c.chunkSize ||
				c.cells[j].pos < c.curPositions[i]-c.swaWindowSize {
				mask[i*length+(j-c.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	maskTensor := ctx.Input().FromFloats(mask, length, c.curBatchSize)

	if c.config.MaskDType != ml.DTypeF32 {
		maskTensor = maskTensor.Cast(ctx, c.config.MaskDType)
	}

	return maskTensor
}
