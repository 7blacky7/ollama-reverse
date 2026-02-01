// llama_context.go
// Kontext-Modul: Context-Parameter, Context-Management und KV-Cache-Operationen

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams(numCtx int, batchSize int, numSeqMax int, threads int, flashAttention ml.FlashAttentionType, kvCacheType string) ContextParams {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint(numCtx)
	params.n_batch = C.uint(batchSize * numSeqMax)
	params.n_ubatch = C.uint(batchSize)
	params.n_seq_max = C.uint(numSeqMax)
	params.n_threads = C.int(threads)
	params.n_threads_batch = params.n_threads
	params.embeddings = C.bool(true)
	switch flashAttention {
	case ml.FlashAttentionEnabled:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_ENABLED)
	case ml.FlashAttentionDisabled:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_DISABLED)
	case ml.FlashAttentionAuto:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_AUTO)
	}
	params.type_k = kvCacheTypeFromStr(strings.ToLower(kvCacheType))
	params.type_v = kvCacheTypeFromStr(strings.ToLower(kvCacheType))

	return ContextParams{c: params}
}

type Context struct {
	c          *C.struct_llama_context
	numThreads int
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

func (c *Context) Decode(batch *Batch) error {
	// Positive return values does not mean a fatal error, but rather a warning.
	//   0 - success
	//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
	// < 0 - error
	code := int(C.llama_decode(c.c, batch.c))

	if code < 0 {
		return fmt.Errorf("llama_decode failed with code %d", code)
	}

	if code > 0 {
		return ErrKvCacheFull
	}

	return nil
}

func (c *Context) Model() *Model {
	return &Model{c: C.llama_get_model(c.c)}
}

func (c *Context) KvCacheSeqAdd(seqId int, p0 int, p1 int, delta int) {
	C.llama_memory_seq_add(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1), C.int(delta))
}

func (c *Context) KvCacheSeqRm(seqId int, p0 int, p1 int) bool {
	return bool(C.llama_memory_seq_rm(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1)))
}

func (c *Context) KvCacheSeqCp(srcSeqId int, dstSeqId int, p0 int, p1 int) {
	C.llama_memory_seq_cp(C.llama_get_memory(c.c), C.int(srcSeqId), C.int(dstSeqId), C.int(p0), C.int(p1))
}

func (c *Context) KvCacheClear() {
	C.llama_memory_clear(C.llama_get_memory(c.c), true)
}

func (c *Context) KvCacheCanShift() bool {
	return bool(C.llama_memory_can_shift(C.llama_get_memory(c.c)))
}

// Get the embeddings for a sequence id
func (c *Context) GetEmbeddingsSeq(seqId int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_seq(c.c, C.int(seqId)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

func (c *Context) GetEmbeddingsIth(i int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_ith(c.c, C.int32_t(i)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

// GetLogitsIth gets the logits for the ith token
func (c *Context) GetLogitsIth(i int) []float32 {
	logits := unsafe.Pointer(C.llama_get_logits_ith(c.c, C.int32_t(i)))
	if logits == nil {
		return nil
	}

	vocabSize := c.Model().NumVocab()
	result := make([]float32, vocabSize)
	_ = copy(result, unsafe.Slice((*float32)(logits), vocabSize))
	return result
}

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	c := Context{
		c:          C.llama_init_from_model(model.c, params.c),
		numThreads: int(params.c.n_threads),
	}
	if c.c == nil {
		return nil, errors.New("unable to create llama context")
	}

	return &c, nil
}

func (c *Context) Synchronize() {
	C.llama_synchronize(c.c)
}
