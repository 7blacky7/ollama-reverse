// llama_sampling.go
// Sampling-Modul: Sampling-Kontext, Grammar und Schema-zu-Grammar-Konvertierung

package llama

/*
#include <stdlib.h>
#include "llama.h"
#include "sampling_ext.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// sampling
// TODO: this is a temporary wrapper to allow calling C++ code from CGo
type SamplingContext struct {
	c *C.struct_common_sampler
}

type SamplingParams struct {
	TopK           int
	TopP           float32
	MinP           float32
	TypicalP       float32
	Temp           float32
	RepeatLastN    int
	PenaltyRepeat  float32
	PenaltyFreq    float32
	PenaltyPresent float32
	PenalizeNl     bool
	Seed           uint32
	Grammar        string
}

func NewSamplingContext(model *Model, params SamplingParams) (*SamplingContext, error) {
	var cparams C.struct_common_sampler_cparams
	cparams.top_k = C.int32_t(params.TopK)
	cparams.top_p = C.float(params.TopP)
	cparams.min_p = C.float(params.MinP)
	cparams.typical_p = C.float(params.TypicalP)
	cparams.temp = C.float(params.Temp)
	cparams.penalty_last_n = C.int32_t(params.RepeatLastN)
	cparams.penalty_repeat = C.float(params.PenaltyRepeat)
	cparams.penalty_freq = C.float(params.PenaltyFreq)
	cparams.penalty_present = C.float(params.PenaltyPresent)
	cparams.seed = C.uint32_t(params.Seed)

	grammar := C.CString(params.Grammar)
	defer C.free(unsafe.Pointer(grammar))

	cparams.grammar = grammar
	context := &SamplingContext{c: C.common_sampler_cinit(model.c, &cparams)}
	if context.c == nil {
		return nil, errors.New("unable to create sampling context")
	}

	runtime.SetFinalizer(context, func(s *SamplingContext) { C.common_sampler_cfree(s.c) })

	return context, nil
}

func (s *SamplingContext) Reset() {
	C.common_sampler_creset(s.c)
}

func (s *SamplingContext) Sample(llamaContext *Context, idx int) int {
	return int(C.common_sampler_csample(s.c, llamaContext.c, C.int(idx)))
}

func (s *SamplingContext) Accept(id int, applyGrammar bool) {
	C.common_sampler_caccept(s.c, C.llama_token(id), C.bool(applyGrammar))
}

// SchemaToGrammar converts the provided JSON schema to a grammar. It returns
// nil if the provided schema is invalid JSON or an invalid JSON schema.
func SchemaToGrammar(schema []byte) []byte {
	cStr := C.CString(string(schema))
	defer C.free(unsafe.Pointer(cStr))

	// Allocate buffer for grammar based on schema length but with upper bound
	maxLen := max(32768, min(1024*1024, len(schema)*4))
	buf := make([]byte, maxLen)

	// Call C function to convert schema to grammar
	n := C.schema_to_grammar(cStr, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(maxLen))
	if n == 0 {
		// preserve nil
		return nil
	}
	return buf[:n]
}

type TokenData struct {
	ID    int32
	Logit float32
}

type Grammar struct {
	c  *C.struct_llama_grammar
	mu sync.Mutex
}

func NewGrammar(grammar string, vocabIds []uint32, vocabValues []string, eogTokens []int32) *Grammar {
	cGrammar := C.CString(grammar)
	defer C.free(unsafe.Pointer(cGrammar))

	cTokens := make([]C.uint32_t, len(vocabIds))
	for i, token := range vocabIds {
		cTokens[i] = C.uint32_t(token)
	}

	cPieces := make([]*C.char, len(vocabValues))
	for i, piece := range vocabValues {
		cPieces[i] = C.CString(piece)
		defer C.free(unsafe.Pointer(cPieces[i]))
	}

	cEogTokens := make([]C.uint32_t, len(eogTokens))
	for i, token := range eogTokens {
		cEogTokens[i] = C.uint32_t(token)
	}

	g := C.grammar_init(cGrammar, unsafe.SliceData(cTokens), C.size_t(len(cTokens)), unsafe.SliceData(cPieces), unsafe.SliceData(cEogTokens), C.size_t(len(cEogTokens)))
	if g == nil {
		return nil
	}

	return &Grammar{c: g}
}

func (g *Grammar) Free() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.c != nil {
		C.grammar_free(g.c)
		g.c = nil
	}
}

func (g *Grammar) Apply(tokens []TokenData) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.c == nil {
		return
	}

	tds := make([]C.struct_llama_token_data, len(tokens))
	for i, token := range tokens {
		tds[i] = C.struct_llama_token_data{
			id:    C.int32_t(token.ID),
			logit: C.float(token.Logit),
			p:     C.float(0.0),
		}
	}
	tda := &C.llama_token_data_array{
		data:     (*C.struct_llama_token_data)(unsafe.Pointer(&tds[0])),
		size:     C.size_t(len(tokens)),
		selected: C.int64_t(-1),
		sorted:   C.bool(false),
	}
	var pinner runtime.Pinner
	pinner.Pin(&tds[0])
	defer pinner.Unpin()

	C.grammar_apply(g.c, tda)
	for i := range tokens {
		tokens[i].Logit = float32(tds[i].logit)
	}
}

func (g *Grammar) Accept(token int32) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if grammar was freed
	if g.c == nil {
		return
	}

	C.grammar_accept(g.c, C.llama_token(token))
}
