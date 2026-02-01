// Package kvcache - Typen und Datenstrukturen (Legacy/Kommentiert)
//
// Dieses Modul enthaelt die auskommentierten Typdefinitionen
// fuer den experimentellen KV-Cache mit erweiterten Head-Dimensionen.
// Die zusaetzlichen Maps (kHeadDims, vHeadDims, numKVHeads) ermoeglichen
// flexiblere Tensor-Konfigurationen pro Layer.
package kvcache

// import (
// 	"github.com/ollama/ollama/ml"
// )

// type shiftFn func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error)

// // Causal cache stores K and V tensors according to their position in the
// // sequence. Returns the history and a mask for attending to past tokens
// //
// // The tensors are of shape embed dim, kv heads, batch size
// // The mask is of shape history size, batch size
// type Causal struct {
// 	DType ml.DType

// 	// swaWindowSize is the number of tokens that will be included in the mask
// 	// during attention operations. swaMemorySize is the number of tokens that
// 	// will be retained in memory for partial prefix caching. Set to math.MaxInt32
// 	// for unlimited or if sliding window attention is not being used.
// 	swaWindowSize int32
// 	swaMemorySize int32

// 	chunkSize int32

// 	opts CausalOptions

// 	// maxBatch is the largest batch that we might receive
// 	maxBatch int

// 	// config controls mostly backend-specific optimizations
// 	config *ml.CacheConfig

// 	// ** current forward pass **

// 	// size of the current batch
// 	curBatchSize int

// 	// locations for data storage for this batch
// 	curLoc ml.Tensor

// 	// mask of the cache as used by this batch
// 	curMask ml.Tensor

// 	// the active layer for Get and Put
// 	curLayer int

// 	// locations in the cache that are needed for this batch
// 	curCellRange cellRange

// 	// curSequences is the sequences corresponding to this pass's entries in the cache
// 	curSequences []int

// 	// curPositions is the positions corresponding to this pass's entries in the cache
// 	curPositions []int32

// 	// ** cache metadata **

// 	// for each possible location in the cache, stores the position and set of sequences
// 	// that reference the data there
// 	cells []cacheCell

// 	// maps from sequence to the range of locations where it is stored in the cache
// 	cellRanges map[int]cellRange

// 	// ** cache data storage **

// 	shiftFn      shiftFn
// 	backend      ml.Backend
// 	ctxs         map[int]ml.Context
// 	keys, values map[int]ml.Tensor

// 	kHeadDims, vHeadDims, numKVHeads map[int]int
// }

// type cacheCell struct {
// 	pos       int32
// 	sequences []int
// }

// type cellRange struct {
// 	min int
// 	max int
// }

// type CausalOptions struct {
// 	// Enabled controls whether the causal mask is generated for a particular index in a batch
// 	Except []int
// }
