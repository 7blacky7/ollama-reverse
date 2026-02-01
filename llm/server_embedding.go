// Package llm - Embedding und Tokenization
//
// Funktionen für Embedding-Generierung und Token-Verarbeitung:
// - Embedding: Text zu Vektor
// - Tokenize: Text zu Token-IDs
// - Detokenize: Token-IDs zu Text
// - Implementierungen für llamaServer und ollamaServer
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// EmbeddingRequest für Embedding-Anfragen
type EmbeddingRequest struct {
	Content string `json:"content"`
}

// EmbeddingResponse mit Embedding-Vektor
type EmbeddingResponse struct {
	Embedding       []float32 `json:"embedding"`
	PromptEvalCount int       `json:"prompt_eval_count"`
}

// Embedding generiert einen Vektor für den Input-Text
func (s *llmServer) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	logutil.Trace("embedding request", "input", input)

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embedding request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return nil, 0, err
	}
	defer s.sem.Release(1)

	// Server-Status prüfen
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return nil, 0, err
	} else if status != ServerStatusReady {
		return nil, 0, fmt.Errorf("unexpected server status: %s", status)
	}

	data, err := json.Marshal(EmbeddingRequest{Content: input})
	if err != nil {
		return nil, 0, fmt.Errorf("error marshaling embed data: %w", err)
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/embedding", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, 0, fmt.Errorf("error creating embed request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, 0, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm embedding error: %s", body)
		return nil, 0, api.StatusError{
			StatusCode:   resp.StatusCode,
			ErrorMessage: string(body),
		}
	}

	var e EmbeddingResponse
	if err := json.Unmarshal(body, &e); err != nil {
		return nil, 0, fmt.Errorf("unmarshal tokenize response: %w", err)
	}

	return e.Embedding, e.PromptEvalCount, nil
}

// Tokenize konvertiert Text zu Token-IDs (llamaServer)
func (s *llamaServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	s.llamaModelLock.Lock()
	defer s.llamaModelLock.Unlock()

	if s.llamaModel == nil {
		return nil, fmt.Errorf("no tokenizer configured")
	}

	return s.llamaModel.Tokenize(content, false, true)
}

// Tokenize konvertiert Text zu Token-IDs (ollamaServer)
func (s *ollamaServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	tokens, err := s.textProcessor.Encode(content, false)
	if err != nil {
		return nil, err
	}

	toks := make([]int, len(tokens))
	for i, t := range tokens {
		toks[i] = int(t)
	}

	return toks, nil
}

// Detokenize konvertiert Token-IDs zurück zu Text (llamaServer)
func (s *llamaServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	s.llamaModelLock.Lock()
	defer s.llamaModelLock.Unlock()

	if s.llamaModel == nil {
		return "", fmt.Errorf("no tokenizer configured")
	}

	var resp string
	for _, token := range tokens {
		resp += s.llamaModel.TokenToPiece(token)
	}

	return resp, nil
}

// Detokenize konvertiert Token-IDs zurück zu Text (ollamaServer)
func (s *ollamaServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	toks := make([]int32, len(tokens))
	for i, t := range tokens {
		toks[i] = int32(t)
	}

	content, err := s.textProcessor.Decode(toks)
	if err != nil {
		return "", err
	}

	return content, nil
}
