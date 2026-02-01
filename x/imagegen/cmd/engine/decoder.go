//go:build mlx

// decoder.go - Decoder für autoregressive Generierung
// Enthält: Decoder Struct, prefill und step Methoden
package main

import (
	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Decoder wraps model + cache for autoregressive generation.
type Decoder struct {
	model         Model
	caches        []cache.Cache
	vocabSize     int32
	temp          float32
	topK          int
	topP          float32
	token         *mlx.Array   // Current token (kept across pools)
	oldCacheState []*mlx.Array // Preallocated slice for old cache state
	image         *mlx.Array   // Optional image for multimodal prefill
}

func NewDecoder(m Model, temp float32, topK int, topP float32) *Decoder {
	caches := m.NewCache(0)
	return &Decoder{
		model:         m,
		caches:        caches,
		vocabSize:     m.VocabSize(),
		temp:          temp,
		topK:          topK,
		topP:          topP,
		oldCacheState: make([]*mlx.Array, 0, len(caches)*2),
	}
}

// SetImage sets the image for multimodal prefill (call before prefill)
func (d *Decoder) SetImage(img *mlx.Array) {
	d.image = img
}

// prefill verarbeitet die Eingabe-Tokens und initialisiert den Cache
func (d *Decoder) prefill(inputIDs []int32) int {
	processed := 0

	// Track old cache state to free after each chunk
	var oldCacheState []*mlx.Array

	// For multimodal models with an image, we need to process all tokens together
	// in the first forward pass so the image embeddings can be inserted properly.
	// Skip chunking for multimodal prefill.
	isMultimodal := d.image != nil

	// Process all-but-1 tokens in chunks, eval cache state for memory management
	// Skip chunking for multimodal - process everything in the final step
	if !isMultimodal {
		for len(inputIDs) > 1 {
			chunkSize := min(2048, len(inputIDs)-1)
			if chunkSize <= 0 {
				break
			}
			chunk := inputIDs[:chunkSize]

			// Save old cache state before forward
			oldCacheState = oldCacheState[:0]
			for _, c := range d.caches {
				oldCacheState = append(oldCacheState, c.State()...)
			}

			var cacheState []*mlx.Array
			withStream(func() {
				x := mlx.NewArrayInt32(chunk, []int32{1, int32(len(chunk))})
				d.model.Forward(x, d.caches)
				for _, c := range d.caches {
					cacheState = append(cacheState, c.State()...)
				}
			})
			mlx.Eval(cacheState...)

			// Free old cache state
			for _, arr := range oldCacheState {
				if arr != nil {
					arr.Free()
				}
			}

			inputIDs = inputIDs[chunkSize:]
			processed += chunkSize
		}
	}

	// Save old cache state before final step
	oldCacheState = oldCacheState[:0]
	for _, c := range d.caches {
		oldCacheState = append(oldCacheState, c.State()...)
	}

	// Final token + sampling (or all tokens for multimodal)
	withStream(func() {
		x := mlx.NewArrayInt32(inputIDs, []int32{1, int32(len(inputIDs))})
		mlx.Eval(x) // Materialize before any other evals

		var logits *mlx.Array
		// Use ForwardWithImage if we have an image and model supports it
		if d.image != nil {
			if mm, ok := d.model.(MultimodalModel); ok {
				logits = mm.ForwardWithImage(x, d.image, d.caches)
				d.image = nil // Only use image for first forward
			} else {
				logits = d.model.Forward(x, d.caches)
			}
		} else {
			logits = d.model.Forward(x, d.caches)
		}
		d.token = sample(logits, d.temp, d.topK, d.topP, d.vocabSize)
	})
	// Keep cache state (token auto-kept by AsyncEval)
	for _, c := range d.caches {
		mlx.Keep(c.State()...)
	}
	mlx.AsyncEval(d.token)

	// Free old cache state from before final step
	for _, arr := range oldCacheState {
		if arr != nil {
			arr.Free()
		}
	}

	mlx.ClearCache()

	return processed + len(inputIDs)
}

// step führt einen einzelnen Generierungsschritt durch
func (d *Decoder) step() int32 {
	prevToken := d.token

	// Save old cache state (reuse preallocated slice)
	d.oldCacheState = d.oldCacheState[:0]
	for _, c := range d.caches {
		d.oldCacheState = append(d.oldCacheState, c.State()...)
	}

	withStream(func() {
		logits := d.model.Forward(mlx.Reshape(prevToken, 1, 1), d.caches)
		d.token = sample(logits, d.temp, d.topK, d.topP, d.vocabSize)
	})
	// Keep token and new cache state so they survive cleanup
	mlx.Keep(d.token)
	for _, c := range d.caches {
		mlx.Keep(c.State()...)
	}
	mlx.AsyncEval(d.token)

	// Sync on previous token (GPU already working on next step)
	val := prevToken.ItemInt32()

	// Free old token and old cache state
	prevToken.Free()
	for _, arr := range d.oldCacheState {
		arr.Free()
	}
	return val
}
