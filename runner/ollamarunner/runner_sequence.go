// runner_sequence.go - Sequenz-Verwaltung fuer den Ollama Runner
//
// Enthaelt:
// - NewSequence: Erstellt neue Inferenz-Sequenzen
// - inputs: Verarbeitet Prompts und Bilder zu Inputs
// - errorInputTooLong: Fehler fuer zu lange Eingaben

package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"runtime"
	"strconv"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// errorInputTooLong wird zurueckgegeben wenn die Eingabe zu lang ist
var errorInputTooLong = errors.New("the input length exceeds the context length")

// NewSequence erstellt eine neue Sequenz fuer Textgenerierung oder Embeddings
func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	inputs, ctxs, mmStore, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = int32(len(inputs))
	}

	// Mindestens 1 Input muss beim Shift verworfen werden koennen
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	if int32(len(inputs)) > s.cache.numCtx {
		if !params.truncate {
			return nil, errorInputTooLong
		}

		discard := int32(len(inputs)) - s.cache.numCtx
		promptStart := params.numKeep + discard

		// Falls wir mitten in einem untrennbaren Batch abschneiden muessen, entferne den ganzen Batch
		sameBatch := 0
		for i, inp := range inputs {
			if sameBatch > 0 {
				sameBatch--

				if promptStart == int32(i) {
					promptStart++
				}
			} else if promptStart == int32(i) {
				break
			}

			if inp.SameBatch != 0 {
				if int32(i) < params.numKeep {
					return nil, fmt.Errorf("SameBatch may not be specified within numKeep (index: %v numKeep: %v SameBatch: %v)", i, params.numKeep, inp.SameBatch)
				}

				sameBatch = inp.SameBatch
			}
		}

		if promptStart >= int32(len(inputs)) {
			return nil, errors.New("entire prompt removed by truncation")
		}

		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[promptStart:]...)

		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	// TODO(jessegross): Ingest cached history for grammar

	return &Sequence{
		ctxs:             ctxs,
		mmStore:          mmStore,
		inputs:           inputs,
		numPromptInputs:  len(inputs),
		numPredict:       params.numPredict,
		pendingResponses: make([]string, 0),
		responses:        make(chan response, 100),
		quit:             make(chan bool, 1),
		embedding:        make(chan []float32, 1),
		sampler:          params.sampler,
		embeddingOnly:    params.embedding,
		stop:             params.stop,
		numKeep:          params.numKeep,
		shift:            params.shift,
		logprobs:         params.logprobs,
		topLogprobs:      params.topLogprobs,
	}, nil
}

// inputs verarbeitet Prompt und Bilder zu einer Liste von Inputs
// Der Prompt wird an [img-<n>] Tags aufgeteilt, Text wird tokenisiert, Bilder decodiert
func (s *Server) inputs(prompt string, images []llm.ImageData) ([]*input.Input, []ml.Context, multimodalStore, error) {
	var inputs []*input.Input
	var ctxs []ml.Context
	var mmStore multimodalStore

	var parts []string
	var matches [][]string

	multimodalProcessor, visionModel := s.model.(model.MultimodalProcessor)

	if visionModel {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
		mmStore = newMultimodalStore()
	} else {
		parts = []string{prompt}
	}

	for i, part := range parts {
		// Text tokenisieren
		tokens, err := s.model.(model.TextProcessor).Encode(part, i == 0)
		if err != nil {
			return nil, nil, nil, err
		}

		for _, t := range tokens {
			inputs = append(inputs, &input.Input{Token: t})
		}

		// Bild decodieren und speichern
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
				return nil, nil, nil, fmt.Errorf("invalid image index: %d", n)
			}

			ctx := s.model.Backend().NewContext()
			runtime.SetFinalizer(ctx, func(c ml.Context) { c.Close() })
			ctxs = append(ctxs, ctx)
			imageEmbeddings, err := multimodalProcessor.EncodeMultimodal(ctx, images[imageIndex].Data)
			if err != nil {
				return nil, nil, nil, err
			}

			s.multimodalHash.Reset()
			_, _ = s.multimodalHash.Write(images[imageIndex].Data)
			imageHash := s.multimodalHash.Sum64()

			mmStore.addMultimodal(imageEmbeddings)

			inputs = append(inputs, &input.Input{Multimodal: imageEmbeddings, MultimodalHash: imageHash})
		}
	}

	if visionModel {
		var err error
		inputs, err = multimodalProcessor.PostTokenize(inputs)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	return inputs, ctxs, mmStore, nil
}
