// runner_batch.go - Batch-Verarbeitung fuer den Ollama Runner
//
// Enthaelt:
// - run: Haupt-Loop fuer Batch-Verarbeitung
// - forwardBatch: Berechnet einen Batch
// - computeBatch: Asynchrone Verarbeitung eines Batches
// - flushPending: Sendet ausstehende Antworten
// - removeSequence: Entfernt eine Sequenz
// - allNil: Prueft ob alle Sequenzen nil sind
// - calculateLogprobs: Berechnet Log-Wahrscheinlichkeiten

package ollamarunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/runner/common"
)

// allNil prueft ob alle Sequenzen nil sind
func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

// flushPending sendet alle ausstehenden Antworten einer Sequenz
func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	logprobs := seq.pendingLogprobs
	seq.pendingResponses = []string{}
	seq.pendingLogprobs = []llm.Logprob{}

	// Pruefe auf unvollstaendige UTF-8 Zeichen
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- response{content: joined, logprobs: logprobs}:
		return true
	case <-seq.quit:
		return false
	}
}

// removeSequence entfernt eine Sequenz und gibt Ressourcen frei
func (s *Server) removeSequence(seqIndex int, reason llm.DoneReason) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
	s.seqsSem.Release(1)
}

// calculateLogprobs konvertiert Logits zu Log-Wahrscheinlichkeiten und findet Top-K Tokens
func calculateLogprobs(logits []float32, selectedToken int32, topK int, textProcessor model.TextProcessor) []llm.Logprob {
	decoder := func(tokenID int) string {
		text, _ := textProcessor.Decode([]int32{int32(tokenID)})
		return text
	}
	return common.CalculateLogprobs(logits, int(selectedToken), topK, decoder)
}

// run ist der Haupt-Loop fuer die Batch-Verarbeitung
func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	supportsAsync := pooling.Type(s.model.Backend().Config().Uint("pooling_type")) == pooling.TypeNone

	var previousBatch batchState
	for {
		select {
		case <-ctx.Done():
			return
		default:
			var err error
			nextBatch, err := s.forwardBatch(previousBatch)
			if err != nil {
				panic(err)
			}

			if supportsAsync {
				go s.computeBatch(nextBatch)
			} else {
				s.computeBatch(nextBatch)
			}

			previousBatch = nextBatch
		}
	}
}

// forwardBatch berechnet einen Batch und bereitet den naechsten vor
func (s *Server) forwardBatch(pendingBatch batchState) (nextBatch batchState, err error) {
	// Warte auf den vorherigen Batch falls vorhanden
	if pendingBatch.ctx != nil {
		logutil.Trace("forwardBatch waiting for compute to start", "pendingBatch.id", pendingBatch.id)
		<-pendingBatch.computeStartedCh
		logutil.Trace("forwardBatch compute started, setting up next batch", "pendingBatch.id", pendingBatch.id, "id", s.batchID)
		nextBatch.inputsReadyCh = pendingBatch.outputsReadyCh
	} else {
		logutil.Trace("forwardBatch no pending batch detected", "batchID", s.batchID)
		nextBatch.inputsReadyCh = make(chan struct{}, 1)
		nextBatch.inputsReadyCh <- struct{}{}
	}

	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait()
	}
	defer s.mu.Unlock()

	nextBatch.ctx = s.model.Backend().NewContext()
	defer func() {
		if err != nil {
			nextBatch.ctx.Close()
			nextBatch.ctx = nil
		}
	}()
	nextBatch.id = s.batchID
	nextBatch.seqs = append([]*Sequence{}, s.seqs...)
	nextBatch.computeStartedCh = make(chan struct{}, 1)
	nextBatch.outputsReadyCh = make(chan struct{}, 1)

	var batchInputs []*input.Input
	var batchOutputs []int32
	var batch input.Batch

	resumeSeq := -1
	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]
		if seq == nil {
			continue
		}

		// Pruefe numPredict Limit
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(seqIdx, llm.DoneReasonLength)
			nextBatch.seqs[seqIdx] = nil
			continue
		}

		if !s.cache.enabled {
			seq.inputs = append(seq.cache.Inputs, seq.inputs...)
			seq.cache.Inputs = []*input.Input{}
		}

		batchSize := s.batchSize

		for i, inp := range seq.inputs {
			minBatch := 1 + inp.SameBatch
			if minBatch > batchSize {
				batchSize = minBatch
			}

			if len(batchInputs)+minBatch > batchSize {
				if len(seq.pendingInputs) == 0 && resumeSeq == -1 {
					resumeSeq = seqIdx
				}
				break
			}

			if int32(len(seq.cache.Inputs)+len(seq.pendingInputs)+minBatch) > s.cache.numCtx {
				if len(seq.pendingInputs) != 0 {
					break
				}

				if !seq.shift {
					s.removeSequence(seqIdx, llm.DoneReasonLength)
					nextBatch.seqs[seqIdx] = nil
					break
				}

				err = s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
				if err != nil {
					var reprocess *ErrReprocessInputs
					if errors.As(err, &reprocess) {
						seq.inputs = append(reprocess.Inputs, seq.inputs...)
						nextBatch.seqs[seqIdx] = nil
						err = nil
						continue
					} else {
						return
					}
				}
			}

			batchInputs = append(batchInputs, seq.inputs[i])
			if inp.Multimodal != nil {
				var mm []input.Multimodal
				mm, err = seq.mmStore.getMultimodal(s.model.Backend(), nextBatch.ctx, inp.Multimodal, false)
				if err != nil {
					return
				}
				batch.Multimodal = append(batch.Multimodal, input.MultimodalIndex{Index: len(batchInputs) - 1, Multimodal: mm})
			}

			batch.Positions = append(batch.Positions, int32(len(seq.cache.Inputs)+len(seq.pendingInputs)))
			batch.Sequences = append(batch.Sequences, seq.cache.Id)

			seq.iBatch = len(batchOutputs)
			if i+1 == len(seq.inputs) || seq.embeddingOnly {
				batchOutputs = append(batchOutputs, int32(len(batchInputs)-1))
			}
			logutil.Trace("forwardBatch iBatch", "batchID", s.batchID, "seqIdx", seqIdx, "seq.iBatch", seq.iBatch, "i+1", i+1, "len(seq.inputs)", len(seq.inputs))
			seq.pendingInputs = append(seq.pendingInputs, inp)
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	startedAt := time.Now()
	for i := range nextBatch.seqs {
		if nextBatch.seqs[i] != nil && nextBatch.seqs[i].startedAt.IsZero() {
			nextBatch.seqs[i].startedAt = startedAt
		}
	}

	if resumeSeq != -1 {
		s.nextSeq = resumeSeq
	} else {
		s.nextSeq = seqIdx + 1
	}

	if len(batchInputs) == 0 {
		logutil.Trace("forwardBatch no batchInputs, going idle", "batchID", s.batchID)
		nextBatch.ctx.Close()
		nextBatch.ctx = nil
		return
	}
	s.batchID++

	batch.Inputs = nextBatch.ctx.Input().Empty(ml.DTypeI32, len(batchInputs))
	batch.Outputs = nextBatch.ctx.Input().FromInts(batchOutputs, len(batchOutputs))
	nextBatch.ctx.SetBatchSize(len(batchInputs))
	nextBatch.modelOutput, err = model.Forward(nextBatch.ctx, s.model, batch)
	if err != nil {
		err = fmt.Errorf("failed to build graph: %w", err)
		return
	}
	nextBatch.batchInputs = batchInputs
	nextBatch.batch = batch

	return
}
