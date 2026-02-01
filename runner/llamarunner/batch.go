// Package llamarunner - Batch-Verarbeitung
//
// Dieses Modul enthält die Batch-Verarbeitungslogik:
// - run: Hauptschleife für kontinuierliche Batch-Verarbeitung
// - processBatch: Verarbeitet einen Batch von Inputs
// - calculateLogprobsLlama: Berechnet Log-Wahrscheinlichkeiten
package llamarunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/runner/common"
)

// calculateLogprobsLlama konvertiert rohe Logits zu Log-Wahrscheinlichkeiten
// und findet die Top-K Tokens
func calculateLogprobsLlama(logits []float32, selectedToken int, topK int, model *llama.Model) []llm.Logprob {
	return common.CalculateLogprobs(logits, selectedToken, topK, model.TokenToPiece)
}

// run ist die Hauptschleife für die kontinuierliche Batch-Verarbeitung
func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	// Batches werden hier einmalig allokiert für bessere Performance
	tokenBatch, err := llama.NewBatch(s.batchSize, len(s.seqs), 0)
	if err != nil {
		panic(err)
	}
	defer tokenBatch.Free()

	var embedBatch *llama.Batch
	embedBatchSize := s.image.BatchSize(s.batchSize)
	if embedBatchSize != 0 {
		embedBatch, err = llama.NewBatch(embedBatchSize, len(s.seqs), s.image.EmbedSize(s.lc))
		if err != nil {
			panic(err)
		}
		defer embedBatch.Free()
	} else {
		embedBatch = &llama.Batch{}
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
			err := s.processBatch(tokenBatch, embedBatch)
			if err != nil {
				panic(err)
			}

			tokenBatch.Clear()
			embedBatch.Clear()
		}
	}
}

// processBatch verarbeitet einen Batch von Token- oder Embedding-Inputs.
// TODO (jmorganca): processBatch sollte vereinfacht werden durch Entfernen von:
// * Sampling
// * Stop-Token-Prüfung
// * Metriken
// Diese sollten stattdessen von den Handlern verarbeitet werden.
// processBatch sollte nur für das Annehmen von Tokens/Embeddings
// und schnellstmögliche Batch-Verarbeitung zuständig sein.
func (s *Server) processBatch(tokenBatch *llama.Batch, embedBatch *llama.Batch) error {
	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Warten bis ein Element hinzugefügt wird
	}
	defer s.mu.Unlock()

	var batch *llama.Batch
	var numOutputs int

	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]

		if seq == nil {
			continue
		}

		// Prüfen ob numPredict-Limit überschritten
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(seqIdx, llm.DoneReasonLength)
			continue
		}

		for i, input := range seq.inputs {
			if len(seq.cache.Inputs)+len(seq.pendingInputs)+1 > s.cache.numCtx {
				if len(seq.pendingInputs) == 0 {
					if !seq.shift {
						s.removeSequence(seqIdx, llm.DoneReasonLength)
						break
					}

					err := s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
					if err != nil {
						var reprocess *ErrReprocessInputs
						if errors.As(err, &reprocess) {
							// Diese Inputs zur Neuverarbeitung an die Queue anhängen
							seq.inputs = append(reprocess.Inputs, seq.inputs...)
							// Normal weiterverarbeiten
							continue
						} else {
							return err
						}
					}
				} else {
					break
				}
			}

			embedding := input.embed != nil

			// Falls noch kein Batch: den korrekten Typ wählen und so voll wie
			// möglich über alle Sequenzen füllen. Bei Input des anderen Typs
			// für diese Sequenz stoppen, aber beim nächsten Batch dort weitermachen
			// um Typen zu alternieren.
			if batch == nil {
				if !embedding {
					batch = tokenBatch
				} else {
					batch = embedBatch
				}
			} else if embedding != batch.IsEmbedding() {
				s.nextSeq = seqIdx
				break
			}

			if i >= batch.Size() {
				break
			}

			output := i+1 == len(seq.inputs)
			batch.Add(input.token, input.embed, len(seq.cache.Inputs)+len(seq.pendingInputs), output, seq.cache.Id)
			if output {
				numOutputs++
			}

			seq.pendingInputs = append(seq.pendingInputs, input)
			seq.iBatch = batch.NumTokens() - 1
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	if batch == nil || batch.NumTokens() == 0 {
		return nil
	}

	t := time.Now()
	if err := s.lc.Decode(batch); err != nil {
		return fmt.Errorf("failed to decode batch: %w", err)
	}

	if numOutputs > 0 {
		s.lc.Synchronize()
	}

	// Nach dem Dekodieren: pendingInputs sind jetzt im Cache
	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []input{}
		}

		// Kein Sampling während Prompt-Verarbeitung
		if len(seq.inputs) != 0 {
			seq.processingDuration += time.Since(t)
			continue
		}

		seq.numDecoded++
		if seq.numDecoded > 1 {
			seq.generationDuration += time.Since(t)
		} else {
			seq.processingDuration += time.Since(t)
		}

		// Falls Prompt fertig verarbeitet: Embedding generieren und zurückgeben
		if seq.embeddingOnly {
			embed := s.lc.GetEmbeddingsSeq(seq.cache.Id)
			if embed == nil {
				embed = s.lc.GetEmbeddingsIth(seq.iBatch)
			}

			seq.embedding <- embed
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		// Token samplen
		token := seq.samplingCtx.Sample(s.lc, seq.iBatch)
		seq.samplingCtx.Accept(token, true)
		piece := s.model.TokenToPiece(token)

		seq.numPredicted++

		// Bei End-of-Sequence Token abbrechen
		if s.model.TokenIsEog(token) {
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		// Logprobs berechnen falls angefordert (nach EOS-Prüfung)
		if seq.logprobs {
			logits := s.lc.GetLogitsIth(seq.iBatch)
			if logits != nil {
				logprobs := calculateLogprobsLlama(logits, token, seq.topLogprobs, s.model)
				seq.pendingLogprobs = append(seq.pendingLogprobs, logprobs...)
			}
		}

		seq.inputs = []input{{token: token}}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := common.FindStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = common.TruncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Logprobs passend zu den gekürzten Responses kürzen
			if seq.logprobs {
				origLogprobsLen := len(seq.pendingLogprobs)
				numTokensRemoved := origLen - newLen
				newLogprobsLen := origLogprobsLen - numTokensRemoved
				if newLogprobsLen < 0 {
					newLogprobsLen = 0
				}
				seq.pendingLogprobs = seq.pendingLogprobs[:newLogprobsLen]
			}

			// Cache basierend auf zurückgegebenen Tokens aktualisieren:
			// - Wir haben 1 Token mehr als aktuell im Cache (letzter wurde nicht dekodiert)
			// - Entfernte Stop-Sequenzen abziehen
			// - Falls truncateStop einen Teil eines Tokens entfernt hat: diesen droppen
			// - Als Defense-in-Depth: falls truncatedToken kein Stop fand, Extra-Token entfernen
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

	return nil
}
