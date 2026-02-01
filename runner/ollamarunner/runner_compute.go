// runner_compute.go - Asynchrone Batch-Berechnung fuer den Ollama Runner
//
// Enthaelt:
// - computeBatch: Asynchrone Verarbeitung eines Batches

package ollamarunner

import (
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/runner/common"
)

// computeBatch fuehrt die asynchrone Verarbeitung eines Batches durch
func (s *Server) computeBatch(activeBatch batchState) {
	if activeBatch.ctx == nil {
		return
	}
	defer activeBatch.ctx.Close()

	// Warte auf bereite Inputs
	logutil.Trace("computeBatch: waiting for inputs to be ready", "batchID", activeBatch.id)
	<-activeBatch.inputsReadyCh
	logutil.Trace("computeBatch: inputs are ready", "batchID", activeBatch.id)

	// Signalisiere dem naechsten Batch dass Outputs bereit sind
	defer func() {
		logutil.Trace("computeBatch: outputs are ready", "batchID", activeBatch.id)
		activeBatch.outputsReadyCh <- struct{}{}
	}()

	s.mu.Lock()

	// Sammle die tatsaechlichen Input-Token-Werte
	batchInputs := make([]int32, len(activeBatch.batchInputs))
	for i := range batchInputs {
		batchInputs[i] = activeBatch.batchInputs[i].Token
	}

	// Bereite Platzhalter-Tokens fuer den naechsten Batch vor
	nextBatchTokens := make([]*input.Input, len(s.seqs))
	iBatches := make([]int, len(s.seqs))
	for i, seq := range s.seqs {
		iBatches[i] = -1
		if seq == nil {
			continue
		}
		// Ueberspringe neue oder uebersprungene Sequenzen
		if activeBatch.seqs[i] == nil {
			continue
		}

		// Erkenne ob die Sequenz ersetzt wurde
		if seq != activeBatch.seqs[i] {
			logutil.Trace("computeBatch: sequence replaced, discarding its results", "batchID", activeBatch.id, "seqIdx", i)
			continue
		}

		// Pending Inputs kommen nach Compute in den Cache
		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []*input.Input{}
		}

		// Keine Samples waehrend Prompt-Verarbeitung
		if len(seq.inputs) != 0 {
			if !s.cache.enabled {
				panic("caching disabled but unable to fit entire input in a batch")
			}
			continue
		}

		seq.numPredicted++
		nextToken := &input.Input{Token: 0} // Platzhalter
		seq.inputs = []*input.Input{nextToken}
		nextBatchTokens[i] = nextToken
		iBatches[i] = seq.iBatch
	}

	// seqs sind bereit fuer forwardBatch
	s.mu.Unlock()

	activeBatch.batch.Inputs.FromInts(batchInputs)
	activeBatch.ctx.ComputeWithNotify(
		func() {
			logutil.Trace("computeBatch: signaling computeStartedCh", "batchID", activeBatch.id)
			activeBatch.computeStartedCh <- struct{}{}
		},
		activeBatch.modelOutput)

	outputs := activeBatch.modelOutput.Floats()
	t := time.Now()

	logutil.Trace("computeBatch: logits ready", "batchID", activeBatch.id)

	s.mu.Lock()
	defer s.mu.Unlock()

	logutil.Trace("computeBatch: decoding", "batchID", activeBatch.id)
	for i, seq := range s.seqs {
		if seq == nil || nextBatchTokens[i] == nil {
			continue
		}

		seq.lastUpdatedAt = t
		if seq.numPredicted == 1 {
			seq.processingDuration = seq.lastUpdatedAt.Sub(seq.startedAt)
			seq.startedAt = seq.lastUpdatedAt
		}

		// Embedding-Modus: Gib Embedding zurueck
		if seq.embeddingOnly {
			seq.embedding <- outputs
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		// Sample ein Token
		vocabSize := len(outputs) / activeBatch.batch.Outputs.Dim(0)
		logutil.Trace("computeBatch: vocab details", "batchID", activeBatch.id, "seqIdx", i, "len(logits)", len(outputs), "len(activeBatch.batch.Outputs)", activeBatch.batch.Outputs.Dim(0), "vocabSize", vocabSize, "iBatches", iBatches)
		logits := outputs[iBatches[i]*vocabSize : (iBatches[i]+1)*vocabSize]
		token, err := seq.sampler.Sample(logits)
		if err != nil {
			panic("failed to sample token")
		}

		nextBatchTokens[i].Token = token

		// Pruefe auf End-of-Sequence Token
		if s.model.(model.TextProcessor).Is(token, model.SpecialEOS) {
			logutil.Trace("computeBatch: EOS", "batchID", activeBatch.id, "seqIdx", i)
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		piece, err := s.model.(model.TextProcessor).Decode([]int32{token})
		if err != nil {
			panic("failed to decode token")
		}

		// Berechne Logprobs falls angefordert
		if seq.logprobs {
			logprobs := calculateLogprobs(logits, token, seq.topLogprobs, s.model.(model.TextProcessor))
			seq.pendingLogprobs = append(seq.pendingLogprobs, logprobs...)
		}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := common.FindStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = common.TruncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Kuerze Logprobs entsprechend
			if seq.logprobs {
				origLogprobsLen := len(seq.pendingLogprobs)
				numTokensRemoved := origLen - newLen
				newLogprobsLen := origLogprobsLen - numTokensRemoved
				if newLogprobsLen < 0 {
					newLogprobsLen = 0
				}
				seq.pendingLogprobs = seq.pendingLogprobs[:newLogprobsLen]
			}

			// Aktualisiere Cache basierend auf zurueckgegebenen Tokens
			tokenLen := len(seq.cache.Inputs) + 1
			tokenLen -= origLen - newLen
			if tokenTruncated || origLen == newLen {
				tokenLen--
			}

			seq.cache.Inputs = seq.cache.Inputs[:tokenLen]

			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		if common.ContainsStopSuffix(sequence, seq.stop) {
			continue
		}

		if common.IncompleteUnicode(sequence) {
			continue
		}

		if !flushPending(seq) {
			s.removeSequence(i, llm.DoneReasonConnectionClosed)
		}
	}

	samplingDuration := time.Since(t)
	for i, seq := range s.seqs {
		if seq != nil && nextBatchTokens[i] != nil {
			s.seqs[i].samplingDuration += samplingDuration
		}
	}
}
