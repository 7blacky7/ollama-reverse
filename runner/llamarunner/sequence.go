// Package llamarunner - Sequenz-Verwaltung
//
// Dieses Modul enthält die Logik für Sequenz-Operationen:
// - NewSequence: Erstellt eine neue Inferenz-Sequenz
// - inputs: Verarbeitet Prompt und Bilder zu Input-Liste
// - flushPending: Sendet ausstehende Responses
// - removeSequence: Entfernt eine Sequenz aus dem Server
package llamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
)

// errorInputTooLong wird zurückgegeben wenn der Input die Kontextlänge überschreitet
var errorInputTooLong = errors.New("the input length exceeds the context length")

// NewSequence erstellt eine neue Inferenz-Sequenz
func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	inputs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = len(inputs)
	}

	if s.model.AddBOSToken() {
		params.numKeep += 1
	}

	// Sicherstellen dass mindestens 1 Input beim Shift verworfen werden kann
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	if len(inputs) > s.cache.numCtx {
		discard := len(inputs) - s.cache.numCtx
		if !params.truncate {
			return nil, errorInputTooLong
		}

		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[params.numKeep+discard:]...)

		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	var sc *llama.SamplingContext
	if params.samplingParams != nil {
		sc, err = llama.NewSamplingContext(s.model, *params.samplingParams)
		if err != nil {
			return nil, err
		}
		for _, input := range inputs {
			if input.embed == nil {
				sc.Accept(input.token, false)
			}
		}
	}

	return &Sequence{
		inputs:           inputs,
		numPromptInputs:  len(inputs),
		numPredict:       params.numPredict,
		pendingResponses: make([]string, 0),
		responses:        make(chan response, 100),
		quit:             make(chan bool, 1),
		embedding:        make(chan []float32, 1),
		samplingCtx:      sc,
		embeddingOnly:    params.embedding,
		stop:             params.stop,
		numKeep:          params.numKeep,
		shift:            params.shift,
		logprobs:         params.logprobs,
		topLogprobs:      params.topLogprobs,
	}, nil
}

// inputs verarbeitet Prompt und Bilder zu einer Liste von Inputs.
// Der Prompt wird an [img-<n>] Tags gesplittet, Text tokenisiert und
// Bild-Embeddings für jedes Bild generiert.
func (s *Server) inputs(prompt string, images []llm.ImageData) ([]input, error) {
	var inputs []input
	var parts []string
	var matches [][]string

	if s.image != nil {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
	} else {
		parts = []string{prompt}
	}

	for i, part := range parts {
		// Text tokenisieren
		tokens, err := s.lc.Model().Tokenize(part, i == 0, true)
		if err != nil {
			return nil, err
		}

		for _, t := range tokens {
			inputs = append(inputs, input{token: t})
		}

		// Bild - Embedding generieren
		if i < len(matches) {
			n, _ := strconv.Atoi(matches[i][1])

			imageIndex := -1
			for j := range images {
				if images[j].ID == n {
					imageIndex = j
					break
				}
			}

			if imageIndex < 0 {
				return nil, fmt.Errorf("invalid image index: %d", n)
			}

			chunks, err := s.image.MultimodalTokenize(s.lc, images[imageIndex].Data)
			if err != nil {
				return nil, err
			}

			for _, c := range chunks {
				if len(c.Embed) != 0 {
					inputs = append(inputs, input{embed: c.Embed})
				} else {
					for _, t := range c.Tokens {
						inputs = append(inputs, input{token: t})
					}
				}
			}
		}
	}

	return inputs, nil
}

// flushPending sendet alle ausstehenden Responses und gibt true zurück bei Erfolg
func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	logprobs := seq.pendingLogprobs
	seq.pendingResponses = []string{}
	seq.pendingLogprobs = []llm.Logprob{}

	// Prüfen auf unvollständige UTF-8 Zeichen.
	// Wir prüfen bereits während der Generierung, aber einige könnten
	// hier noch ankommen:
	// - Sequenz endet, z.B. Generierungslimit erreicht
	// - Ungültige Zeichen mitten im String
	// Dies ist eine strikte Prüfung um niemals ungültiges Unicode auszugeben.
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

// removeSequence entfernt eine Sequenz aus dem Server und gibt Ressourcen frei
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
