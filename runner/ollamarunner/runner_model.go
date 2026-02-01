// runner_model.go - Model Loading und Konfiguration fuer den Ollama Runner
//
// Enthaelt:
// - allocModel: Alloziert Speicher fuer ein Model
// - closeModel: Gibt Model-Ressourcen frei
// - loadModel: Laedt Model-Gewichte
// - reserveWorstCaseGraph: Reserviert Speicher fuer Worst-Case Graph

package ollamarunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"

	"golang.org/x/image/bmp"
	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// reserveWorstCaseGraph reserviert den maximalen Speicher fuer einen Graph
func (s *Server) reserveWorstCaseGraph(prompt bool) error {
	ctx := s.model.Backend().NewContext()
	defer ctx.Close()

	var err error
	batchSize := 1
	if prompt {
		batchSize = s.batchSize
	}

	inputs := make([]*input.Input, batchSize)
	for i := range inputs {
		inputs[i] = &input.Input{}
	}
	mmStore := newMultimodalStore()

	// Multimodale Strategie:
	// - Encodiere ein 2048x2048 Bild (Worst-Case Annahme)
	// - Fuege Embedding zu einem vollen Batch hinzu
	// - Fuehre PostTokenize aus
	// - Kuerze auf Batch-Groesse
	if multimodalProcessor, ok := s.model.(model.MultimodalProcessor); prompt && ok {
		mmCtx := s.model.Backend().NewContext()
		defer mmCtx.Close()

		img := image.NewGray(image.Rect(0, 0, worstCaseImageSize, worstCaseImageSize))
		var buf bytes.Buffer
		bmp.Encode(&buf, img)

		if inputs[0].Multimodal, err = multimodalProcessor.EncodeMultimodal(mmCtx, buf.Bytes()); err == nil {
			mmStore.addMultimodal(inputs[0].Multimodal)

			inputs, err = multimodalProcessor.PostTokenize(inputs)
			if err != nil {
				return err
			}

			for i, inp := range inputs {
				minBatch := 1 + inp.SameBatch
				if minBatch > s.batchSize {
					inputs = inputs[i:min(i+minBatch, len(inputs))]
					break
				} else if i+minBatch > s.batchSize {
					inputs = inputs[:i]
					break
				}
			}

			if len(inputs) < batchSize {
				newInputs := make([]*input.Input, batchSize)
				copy(newInputs, inputs)
				for i := len(inputs); i < batchSize; i++ {
					newInputs[i] = &input.Input{}
				}
				inputs = newInputs
			}
		}
	}

	var batch input.Batch

	batchInputs := make([]int32, len(inputs))
	batch.Positions = make([]int32, len(inputs))
	batch.Sequences = make([]int, len(inputs))
	for i, inp := range inputs {
		batchInputs[i] = inp.Token
		if inp.Multimodal != nil {
			mm, err := mmStore.getMultimodal(s.model.Backend(), ctx, inp.Multimodal, true)
			if err != nil {
				return err
			}
			batch.Multimodal = append(batch.Multimodal, input.MultimodalIndex{Index: i, Multimodal: mm})
		}

		batch.Positions[i] = int32(i)
	}

	batch.Inputs = ctx.Input().FromInts(batchInputs, len(batchInputs))
	batch.Outputs = ctx.Input().Empty(ml.DTypeI32, s.parallel)

	cache := s.model.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, true)
		if err != nil {
			return err
		}
	}

	t, err := s.model.Forward(ctx, batch)
	if err != nil {
		return err
	}

	ctx.SetBatchSize(batchSize)
	ctx.Forward(t).Reserve()

	return nil
}

// worstCaseImageSize ist die angenommene maximale Bildgroesse fuer Speicherreservierung
const worstCaseImageSize = 2048

// allocModel alloziert den maximalen Speicher fuer ein Model
func (s *Server) allocModel(
	mpath string,
	params ml.BackendParams,
	loraPath []string,
	parallel int,
	kvCacheType string,
	kvSize int,
	multiUserCache bool,
) (panicErr error) {
	// Konvertiere Memory-Allocation Panics zu Errors
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok {
				var noMem ml.ErrNoMem
				if errors.As(err, &noMem) {
					panicErr = noMem
				} else {
					panic(r)
				}
			} else {
				panic(r)
			}
		}
	}()

	var err error
	s.model, err = model.New(mpath, params)
	if err != nil {
		return err
	}

	// TODO(jessegross): LoRA loading
	if len(loraPath) > 0 {
		return errors.New("loras are not yet implemented")
	}

	if s.model.Config().Cache == nil {
		if parallel > 1 {
			parallel = 1
			slog.Warn("model does not support caching, disabling parallel processing")
		}
		if s.batchSize < kvSize {
			s.batchSize = kvSize
			slog.Warn("model does not support caching, setting batch size to context length", "batch_size", kvSize)
		}
	}

	s.cache, err = NewInputCache(s.model, kvCacheType, int32(kvSize), parallel, s.batchSize, multiUserCache)
	if err != nil {
		return err
	}

	s.parallel = parallel
	s.seqs = make([]*Sequence, s.parallel)
	s.seqsSem = semaphore.NewWeighted(int64(s.parallel))

	err = s.reserveWorstCaseGraph(true)
	if err != nil {
		return nil
	}

	return s.reserveWorstCaseGraph(false)
}

// closeModel gibt alle Model-Ressourcen frei
func (s *Server) closeModel() {
	s.cache.Close()
	s.cache = nil
	if s.model != nil {
		s.model.Backend().Close()
		s.model = nil
	}
}

// loadModel laedt die Model-Gewichte (Speicher muss bereits alloziert sein)
func (s *Server) loadModel() {
	err := s.model.Backend().Load(context.TODO(),
		func(progress float32) {
			s.progress = progress
		})
	if err != nil {
		panic(fmt.Errorf("failed to load model: %v", err))
	}

	s.status = llm.ServerStatusReady
	s.ready.Done()
}
