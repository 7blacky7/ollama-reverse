// Package kvcache - Konstruktoren und Initialisierung (Legacy/Kommentiert)
//
// Dieses Modul enthaelt die auskommentierten Factory-Funktionen
// mit erweiterter Head-Dimension-Verwaltung.
// Unterschied zur Produktionsversion: Initialisiert zusaetzlich
// kHeadDims, vHeadDims und numKVHeads Maps.
package kvcache

// import (
// 	"fmt"
// 	"math"

// 	"github.com/ollama/ollama/ml"
// )

// func NewCausalCache(shift shiftFn) *Causal {
// 	return &Causal{
// 		shiftFn:    shift,
// 		ctxs:       make(map[int]ml.Context),
// 		keys:       make(map[int]ml.Tensor),
// 		values:     make(map[int]ml.Tensor),
// 		kHeadDims:  make(map[int]int),
// 		vHeadDims:  make(map[int]int),
// 		numKVHeads: make(map[int]int),
// 	}
// }

// func NewSWACache(windowSize int32, shift shiftFn) *Causal {
// 	return &Causal{
// 		swaWindowSize: windowSize,
// 		shiftFn:       shift,
// 		ctxs:          make(map[int]ml.Context),
// 		keys:          make(map[int]ml.Tensor),
// 		values:        make(map[int]ml.Tensor),
// 		kHeadDims:     make(map[int]int),
// 		vHeadDims:     make(map[int]int),
// 		numKVHeads:    make(map[int]int),
// 	}
// }

// func NewSWAMemCache(windowSize int32, memorySize int32, shift shiftFn) *Causal {
// 	return &Causal{
// 		swaWindowSize: windowSize,
// 		swaMemorySize: memorySize,
// 		shiftFn:       shift,
// 		ctxs:          make(map[int]ml.Context),
// 		keys:          make(map[int]ml.Tensor),
// 		values:        make(map[int]ml.Tensor),
// 		kHeadDims:     make(map[int]int),
// 		vHeadDims:     make(map[int]int),
// 		numKVHeads:    make(map[int]int),
// 	}
// }

// func NewChunkedAttentionCache(chunkSize int32, shift shiftFn) *Causal {
// 	return &Causal{
// 		chunkSize:  chunkSize,
// 		shiftFn:    shift,
// 		ctxs:       make(map[int]ml.Context),
// 		keys:       make(map[int]ml.Tensor),
// 		values:     make(map[int]ml.Tensor),
// 		kHeadDims:  make(map[int]int),
// 		vHeadDims:  make(map[int]int),
// 		numKVHeads: make(map[int]int),
// 	}
// }

// func (c *Causal) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
// 	if c.config == nil {
// 		var config ml.CacheConfig
// 		if cc, ok := backend.(ml.BackendCacheConfig); ok {
// 			config = cc.CacheConfig()
// 		}
// 		c.config = &config
// 	}

// 	if c.config.CachePadding == 0 {
// 		c.config.CachePadding = 1
// 	}

// 	if c.config.MaskBatchPadding == 0 {
// 		c.config.MaskBatchPadding = 1
// 	}

// 	// TODO what types do we handle here?
// 	// if c.config.MaskDType == ml.DTypeOther {
// 	// 	c.config.MaskDType = ml.DTypeFloat32
// 	// }

// 	if c.swaWindowSize == 0 {
// 		c.swaWindowSize = math.MaxInt32
// 	}
// 	if c.swaMemorySize == 0 {
// 		c.swaMemorySize = c.swaWindowSize
// 	}
// 	// We will allocate space in the cache for the stop token, which won't be part of a follow on
// 	// sequence, so allocate an extra token of storage to ensure that we can jump back without
// 	// causing a cache break. As an optimization, only do this when we have parallel sequences
// 	// because the extra token will live in the batch buffer and won't get overwritten if we
// 	// only have a single sequence.
// 	if c.swaMemorySize != math.MaxInt32 && maxSequences > 1 {
// 		c.swaMemorySize = max(c.swaMemorySize, c.swaWindowSize+1)
// 	}
// 	if int(c.swaMemorySize) >= capacity {
// 		c.swaMemorySize = math.MaxInt32
// 	}

// 	if c.swaMemorySize < c.swaWindowSize {
// 		panic(fmt.Errorf("sliding window memory (%v) must be at least as large as the window (%v)", c.swaMemorySize, c.swaWindowSize))
// 	}

// 	var cacheSize int
// 	if c.swaMemorySize == math.MaxInt32 {
// 		cacheSize = maxSequences * capacity
// 	} else {
// 		cacheSize = (maxSequences * int(c.swaMemorySize)) + maxBatch
// 	}
// 	cacheSize = roundUp(cacheSize, c.config.CachePadding)
// 	c.cells = make([]cacheCell, cacheSize)

// 	c.DType = dtype
// 	c.cellRanges = make(map[int]cellRange)
// 	c.backend = backend
// 	c.maxBatch = maxBatch
// }

// func (c *Causal) SetConfig(config ml.CacheConfig) {
// 	if c.config != nil {
// 		panic("config cannot be changed after being previously set, either by the model or backend")
// 	}

// 	c.config = &config
// }

// func (c *Causal) Close() {
// 	slog.Info("XXX Causal.Close called", "number of contexts", len(c.ctxs))
// 	for _, ctx := range c.ctxs {
// 		ctx.Close()
// 	}
// }
