// Package llamarunner - HTTP Handler für Inferenz
//
// Dieses Modul enthält die Request-Handler:
// - completion: Handler für Text-Generierung mit Streaming
// - embeddings: Handler für Embedding-Generierung
// - health: Handler für Gesundheits-Check
package llamarunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
)

// completion verarbeitet Completion-Anfragen und streamt Antworten
func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req llm.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	// Header für Streaming setzen
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Sampling-Parameter aus dem Request extrahieren
	samplingParams := llama.SamplingParams{
		TopK:           req.Options.TopK,
		TopP:           req.Options.TopP,
		MinP:           req.Options.MinP,
		TypicalP:       req.Options.TypicalP,
		Temp:           req.Options.Temperature,
		RepeatLastN:    req.Options.RepeatLastN,
		PenaltyRepeat:  req.Options.RepeatPenalty,
		PenaltyFreq:    req.Options.FrequencyPenalty,
		PenaltyPresent: req.Options.PresencePenalty,
		Seed:           uint32(req.Options.Seed),
		Grammar:        req.Grammar,
	}

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:     req.Options.NumPredict,
		stop:           req.Options.Stop,
		numKeep:        req.Options.NumKeep,
		samplingParams: &samplingParams,
		embedding:      false,
		shift:          req.Shift,
		truncate:       req.Truncate,
		logprobs:       req.Logprobs,
		topLogprobs:    req.TopLogprobs,
	})
	if err != nil {
		if errors.Is(err, errorInputTooLong) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Sicherstellen dass ein Platz für die Sequenz verfügbar ist
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("Failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, true)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}

			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	// Response-Streaming-Schleife
	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case resp, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Content:  resp.content,
					Logprobs: resp.logprobs,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Done:               true,
					DoneReason:         seq.doneReason,
					PromptEvalCount:    seq.numPromptInputs,
					PromptEvalDuration: seq.processingDuration,
					EvalCount:          seq.numDecoded,
					EvalDuration:       seq.generationDuration,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
	}
}

// embeddings verarbeitet Embedding-Anfragen
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req llm.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{
		embedding: true,
		truncate:  false,
	})
	if err != nil {
		if errors.Is(err, errorInputTooLong) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Sicherstellen dass ein Platz für die Sequenz verfügbar ist
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embeddings request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("Failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, false)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}
			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	embedding := <-seq.embedding

	if err := json.NewEncoder(w).Encode(&llm.EmbeddingResponse{
		Embedding:       embedding,
		PromptEvalCount: seq.numPromptInputs,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

// health gibt den aktuellen Server-Status zurück
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&llm.ServerStatusResponse{
		Status:   s.status,
		Progress: s.progress,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}
