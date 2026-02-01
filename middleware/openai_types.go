// openai_types.go - Basis-Typen und Writer-Strukturen fuer OpenAI-Kompatibilitaet
// Enthaelt alle Writer-Typ-Definitionen und die BaseWriter-Implementierung
package middleware

import (
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// BaseWriter ist der Basis-Writer fuer alle OpenAI-kompatiblen Response-Writer
type BaseWriter struct {
	gin.ResponseWriter
}

// ChatWriter verarbeitet Chat-Completion-Responses (streaming und non-streaming)
type ChatWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	toolCallSent  bool
	BaseWriter
}

// CompleteWriter verarbeitet Completion-Responses (streaming und non-streaming)
type CompleteWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	BaseWriter
}

// ListWriter verarbeitet Model-Listen-Responses
type ListWriter struct {
	BaseWriter
}

// RetrieveWriter verarbeitet einzelne Model-Abruf-Responses
type RetrieveWriter struct {
	BaseWriter
	model string
}

// EmbedWriter verarbeitet Embedding-Responses
type EmbedWriter struct {
	BaseWriter
	model          string
	encodingFormat string
}

// writeError konvertiert API-Fehler in OpenAI-kompatibles Fehlerformat
func (w *BaseWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	err := json.Unmarshal(data, &serr)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(http.StatusInternalServerError, serr.Error()))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}
