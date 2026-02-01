// Package llm - Inference (Completion)
//
// Funktionen für Text-Generierung:
// - CompletionRequest/CompletionResponse Typen
// - Completion für streaming Text-Generierung
// - Grammar-Unterstützung für strukturierte Outputs
// - Logprobs für Wahrscheinlichkeits-Ausgabe
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/logutil"
)

// grammarJSON ist die Standard-Grammatik für JSON-Outputs
var grammarJSON = `
root   ::= object
value  ::= object | array | string | number ("true" | "false" | "null") ws
object ::=
  "{" ws (
         string ":" ws value
    ("," ws string ":" ws value)*
  )? ws "}"
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? ws "]"
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\""
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

// maxBufferSize für Scanner-Buffer
const maxBufferSize = 512 * format.KiloByte

// ImageData enthält Bilddaten für multimodale Requests
type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

// CompletionRequest enthält alle Parameter für Text-Generierung
type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Images  []ImageData
	Options *api.Options

	Grammar  string // Wird vor dem Request gesetzt
	Shift    bool
	Truncate bool

	Logprobs    bool // Log-Wahrscheinlichkeiten einschließen
	TopLogprobs int  // Anzahl alternativer Token (0-20)

	// Image Generation Felder
	Width  int32 `json:"width,omitempty"`
	Height int32 `json:"height,omitempty"`
	Steps  int32 `json:"steps,omitempty"`
	Seed   int64 `json:"seed,omitempty"`
}

// DoneReason gibt an warum die Generierung beendet wurde
type DoneReason int

const (
	DoneReasonStop             DoneReason = iota // Natürliches Ende
	DoneReasonLength                             // Längenlimit erreicht
	DoneReasonConnectionClosed                   // Verbindung geschlossen
)

func (d DoneReason) String() string {
	switch d {
	case DoneReasonLength:
		return "length"
	case DoneReasonStop:
		return "stop"
	default:
		return "" // closed
	}
}

// TokenLogprob enthält Log-Wahrscheinlichkeit für ein Token
type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// Logprob enthält vollständige Log-Wahrscheinlichkeits-Info
type Logprob struct {
	TokenLogprob
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

// CompletionResponse ist die Antwort einer Completion
type CompletionResponse struct {
	Content            string        `json:"content"`
	DoneReason         DoneReason    `json:"done_reason"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`

	Logprobs []Logprob `json:"logprobs,omitempty"`

	// Image Generation
	Image      string `json:"image,omitempty"` // Base64-encoded
	Step       int    `json:"step,omitempty"`
	TotalSteps int    `json:"total_steps,omitempty"`
}

// Completion führt eine Text-Generierung durch
func (s *llmServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("completion request", "images", len(req.Images), "prompt", len(req.Prompt), "format", string(req.Format))
	logutil.Trace("completion request", "prompt", req.Prompt)

	if err := s.setupGrammar(&req); err != nil {
		return err
	}

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return err
	}
	defer s.sem.Release(1)

	// NumPredict begrenzen
	if req.Options.NumPredict < 0 || req.Options.NumPredict > 10*s.options.NumCtx {
		req.Options.NumPredict = 10 * s.options.NumCtx
	}

	// Server-Status prüfen
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	return s.executeCompletion(ctx, req, fn)
}

func (s *llmServer) setupGrammar(req *CompletionRequest) error {
	if len(req.Format) == 0 {
		return nil
	}

	switch string(req.Format) {
	case `null`, `""`:
		// Kein Format gesetzt
		return nil
	case `"json"`:
		req.Grammar = grammarJSON
	default:
		if req.Format[0] != '{' {
			return fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", req.Format)
		}

		// User hat JSON Schema angegeben
		g := llama.SchemaToGrammar(req.Format)
		if g == nil {
			return fmt.Errorf("invalid JSON schema in format")
		}
		req.Grammar = string(g)
	}

	return nil
}

func (s *llmServer) executeCompletion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	// JSON marshaling mit unescaped special chars
	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(req); err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating POST request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(serverReq)
	if err != nil && errors.Is(err, context.Canceled) {
		return err
	} else if err != nil {
		slog.Error("post predict", "error", err)
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llm error response: %w", err)
		}
		log.Printf("llm predict error: %s", bodyBytes)
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(bodyBytes))}
	}

	return s.processCompletionStream(ctx, res.Body, fn)
}

func (s *llmServer) processCompletionStream(ctx context.Context, body io.Reader, fn func(CompletionResponse)) error {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(buf, maxBufferSize)

	var lastToken string
	var tokenRepeat int

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}

			var c CompletionResponse
			if err := json.Unmarshal(evt, &c); err != nil {
				return fmt.Errorf("error unmarshalling llm prediction response: %v", err)
			}

			// Token-Wiederholung erkennen
			switch {
			case strings.TrimSpace(c.Content) == lastToken:
				tokenRepeat++
			default:
				lastToken = strings.TrimSpace(c.Content)
				tokenRepeat = 0
			}

			// Abbruch bei zu vielen Wiederholungen
			if tokenRepeat > 30 {
				slog.Debug("prediction aborted, token repeat limit reached")
				return ctx.Err()
			}

			if c.Content != "" {
				fn(CompletionResponse{
					Content:  c.Content,
					Logprobs: c.Logprobs,
				})
			}

			if c.Done {
				fn(c)
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			var msg string
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			} else {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}

		return fmt.Errorf("error reading llm response: %v", err)
	}

	return nil
}
