// openai_chat.go - Chat und Completion Writer-Implementierungen
// Verarbeitet Chat-Completions und Standard-Completions (streaming/non-streaming)
package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// writeResponse verarbeitet Chat-Responses und konvertiert sie ins OpenAI-Format
func (w *ChatWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// Streaming: chat chunk
	if w.stream {
		c := openai.ToChunk(w.id, chatResponse, w.toolCallSent)
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}
		if !w.toolCallSent && len(c.Choices) > 0 && len(c.Choices[0].Delta.ToolCalls) > 0 {
			w.toolCallSent = true
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if chatResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsage(chatResponse)
				c.Usage = &u
				c.Choices = []openai.ChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// Non-streaming: chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer ChatWriter
func (w *ChatWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// writeResponse verarbeitet Generate-Responses und konvertiert sie ins OpenAI-Format
func (w *CompleteWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	err := json.Unmarshal(data, &generateResponse)
	if err != nil {
		return 0, err
	}

	// Streaming: completion chunk
	if w.stream {
		c := openai.ToCompleteChunk(w.id, generateResponse)
		if w.streamOptions != nil && w.streamOptions.IncludeUsage {
			c.Usage = &openai.Usage{}
		}
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if generateResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsageGenerate(generateResponse)
				c.Usage = &u
				c.Choices = []openai.CompleteChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// Non-streaming: completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToCompletion(w.id, generateResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implementiert io.Writer fuer CompleteWriter
func (w *CompleteWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}
