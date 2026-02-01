// llama_vision.go
// Vision-Modul: Multimodal-Verarbeitung (mtmd) fuer Bild-Tokenization

package llama

/*
#include <stdlib.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"unsafe"
)

type MtmdContext struct {
	c *C.struct_mtmd_context
}

func NewMtmdContext(llamaContext *Context, modelPath string) (*MtmdContext, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	// TODO: Support non-default params
	cp := C.mtmd_context_params_default()

	// NOTE: The model and projector embedding lengths are checked during init
	c := C.mtmd_init_from_file(mp, C.llama_get_model(llamaContext.c), cp)
	if c == nil {
		return nil, fmt.Errorf("unable to load mmtd model: %v", modelPath)
	}

	return &MtmdContext{c: c}, nil
}

func (c *MtmdContext) Free() {
	C.mtmd_free(c.c)
}

type MtmdChunk struct {
	Embed  []float32
	Tokens []int
}

func (c *MtmdContext) MultimodalTokenize(llamaContext *Context, data []byte) ([]MtmdChunk, error) {
	// Initialize the input chunks pointer
	ic := C.mtmd_input_chunks_init()
	defer C.mtmd_input_chunks_free(ic)

	// Initialize an empty text prompt so we can tokenize
	it := C.mtmd_input_text_init(C.mtmd_default_marker(), true, true)
	defer C.mtmd_input_text_free(it)

	// Initialize a bitmap with the image data
	bm := C.mtmd_helper_bitmap_init_from_buf(c.c, (*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	defer C.mtmd_bitmap_free(bm)

	// Tokenize the image
	if C.int32_t(0) != C.mtmd_tokenize(c.c, ic, it, &bm, 1) {
		return nil, errors.New("unable to tokenize mtmd embedding from image")
	}
	nChunks := C.mtmd_input_chunks_size(ic)
	numEmbed := llamaContext.Model().NEmbd()
	outChunks := make([]MtmdChunk, 0)
	for i := range int(nChunks) {
		chunk := C.mtmd_input_chunks_get(ic, C.size_t(i))
		numTokens := int(C.mtmd_input_chunk_get_n_tokens(chunk))
		slog.Debug("chunk tokens", "index", i, "numTokens", numTokens)

		if C.mtmd_input_chunk_get_type(chunk) == C.MTMD_INPUT_CHUNK_TYPE_TEXT {
			// If this is a text chunk, add the tokens
			cNumTokens := C.size_t(0)
			cTokens := C.mtmd_input_chunk_get_tokens_text(chunk, &cNumTokens)
			cTokensArr := unsafe.Slice(cTokens, int(cNumTokens))
			tokens := make([]int, int(cNumTokens))
			for j := range int(cNumTokens) {
				tokens[j] = int(cTokensArr[j])
			}
			outChunks = append(outChunks, MtmdChunk{Tokens: tokens})
		} else {
			// Otherwise, encode the image chunk to embeddings

			// Encode the chunk
			if C.int32_t(0) != C.mtmd_encode_chunk(c.c, chunk) {
				return nil, errors.New("unable to encode mtmd image chunk")
			}

			// Get the embeddings for this chunk
			chunkEmbed := make([][]float32, numTokens)
			chunkEmbd := C.mtmd_get_output_embd(c.c)
			if nil == chunkEmbd {
				return nil, errors.New("no mtmd image embedding")
			}

			// Extend the embedding array for each token
			s := unsafe.Slice((*float32)(chunkEmbd), numTokens*numEmbed)
			rows := make([]float32, len(s))
			copy(rows, s)
			for i := range numTokens {
				chunkEmbed[i] = rows[i*numEmbed : (i+1)*numEmbed]
			}
			for _, e := range chunkEmbed {
				outChunks = append(outChunks, MtmdChunk{Embed: e})
			}
		}
	}
	slog.Debug("image tokenization chunks", "totalChunks", len(outChunks))
	return outChunks, nil
}
