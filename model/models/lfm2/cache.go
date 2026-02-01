// Package lfm2 - HybridCache Basisimplementierung
// Dieses Modul enthaelt die HybridCache-Struktur, die einen KV-Cache
// fuer Attention-Layer mit einem rekurrenten Conv-State kombiniert.
package lfm2

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

var _ kvcache.Cache = (*HybridCache)(nil)

// HybridCache stores:
// - a standard causal KV cache for attention layers
// - a per-sequence recurrent conv state for shortconv layers
//
// Conv state shape (per layer, per sequence): [dConv, hiddenSize] where dConv = L_cache - 1.
// Stored internally as a tensor of shape [dConv * hiddenSize, maxSlots].
type HybridCache struct {
	kv *kvcache.Causal

	backend      ml.Backend
	dtype        ml.DType
	maxSequences int

	hiddenSize int
	dConv      int

	// slot mapping for recurrent state
	slotForSeq map[int]int
	refCount   []int
	freeSlots  []int

	// per-layer conv state buffers (allocated lazily)
	convCtxs   map[int]ml.Context
	convStates map[int]ml.Tensor // [dConv*hiddenSize, maxSlots]

	// current forward batch (derived in StartForward)
	curSeqs       []int
	curSlots      []int
	curSlotsInput ml.Tensor
	curSeqTokens  int

	// track if EnsureWritable has been called for this forward pass
	writableEnsured bool
	// track any error from EnsureWritable to propagate later
	writableError error
}

func NewHybridCache(shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error), hiddenSize, dConv int) *HybridCache {
	return &HybridCache{
		kv:         kvcache.NewCausalCache(shift),
		hiddenSize: hiddenSize,
		dConv:      dConv,
		slotForSeq: make(map[int]int),
		convCtxs:   make(map[int]ml.Context),
		convStates: make(map[int]ml.Tensor),
	}
}

func (c *HybridCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.backend = backend
	c.dtype = dtype
	c.maxSequences = maxSequences

	// initialize slot allocator
	c.refCount = make([]int, maxSequences)
	c.freeSlots = c.freeSlots[:0]
	for i := maxSequences - 1; i >= 0; i-- {
		c.freeSlots = append(c.freeSlots, i)
	}

	c.kv.Init(backend, dtype, maxSequences, capacity, maxBatch)
}

func (c *HybridCache) Close() {
	for _, ctx := range c.convCtxs {
		ctx.Close()
	}
	c.kv.Close()
}

func (c *HybridCache) SetConfig(config ml.CacheConfig) {
	c.kv.SetConfig(config)
}

func (c *HybridCache) SetLayer(layer int) {
	c.kv.SetLayer(layer)
}

func (c *HybridCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.kv.Get(ctx)
}

func (c *HybridCache) Put(ctx ml.Context, key, value ml.Tensor) {
	c.kv.Put(ctx, key, value)
}

func (c *HybridCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	if err := c.kv.StartForward(ctx, batch, reserve); err != nil {
		return err
	}

	// Derive equal-length sequence layout for shortconv.
	// LFM2 shortconv assumes tokens form a [seq_tokens, seqs] grid.
	seqCounts := make(map[int]int)
	c.curSeqs = c.curSeqs[:0]
	for _, s := range batch.Sequences {
		if _, ok := seqCounts[s]; !ok {
			c.curSeqs = append(c.curSeqs, s)
		}
		seqCounts[s]++
	}

	if len(c.curSeqs) == 0 {
		return nil
	}

	nTokens := len(batch.Sequences)
	nSeqs := len(c.curSeqs)
	want := nTokens / nSeqs
	for _, s := range c.curSeqs {
		if seqCounts[s] != want {
			return kvcache.ErrNotSupported
		}
	}

	c.curSeqTokens = want

	// When reserving memory for estimation, use fake slot assignments
	// without modifying permanent state (slotForSeq, refCount)
	if reserve {
		c.curSlots = c.curSlots[:0]
		slots := make([]int32, nSeqs)
		for i := range nSeqs {
			c.curSlots = append(c.curSlots, i)
			slots[i] = int32(i)
		}
		c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))
		return nil
	}

	// Ensure slots exist for sequences in this batch
	c.curSlots = c.curSlots[:0]
	var newSlots []int // track newly allocated slots that need zeroing
	for _, s := range c.curSeqs {
		slot, ok := c.slotForSeq[s]
		if !ok {
			var err error
			slot, err = c.allocSlot()
			if err != nil {
				return err
			}
			c.slotForSeq[s] = slot
			c.refCount[slot] = 1
			newSlots = append(newSlots, slot)
		}
		c.curSlots = append(c.curSlots, slot)
	}

	// Zero conv state for newly allocated slots to clear stale data from previous sequences
	if len(newSlots) > 0 {
		c.zeroConvSlots(ctx, newSlots)
	}

	// Create a tensor for the current slots
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	// Reset writable state for new forward pass
	c.writableEnsured = false
	c.writableError = nil

	return nil
}

func (c *HybridCache) CanResume(seq int, pos int32) bool {
	return c.kv.CanResume(seq, pos)
}

func (c *HybridCache) slotsTensor() ml.Tensor {
	return c.curSlotsInput
}

func (c *HybridCache) seqTokens() int {
	return c.curSeqTokens
}

func (c *HybridCache) numSeqs() int {
	return len(c.curSeqs)
}
