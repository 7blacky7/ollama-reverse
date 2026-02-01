// Package model - Registry und Initialisierung.
//
// Diese Datei enthaelt:
// - Modell-Registry (Register, New, NewTextProcessor)
// - modelForArch Helper-Funktion
// - Forward-Funktion fuer Modell-Ausfuehrung
package model

import (
	"errors"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"reflect"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/x/ml"
	_ "github.com/ollama/ollama/x/ml/backend"
	"github.com/ollama/ollama/x/ml/nn/pooling"
	"github.com/ollama/ollama/x/model/input"
)

// ============================================================================
// Fehler-Definitionen
// ============================================================================

var (
	ErrNoVisionModel        = errors.New("this model is missing data required for image input")
	ErrUnsupportedModel     = errors.New("model not supported")
	ErrUnsupportedTokenizer = errors.New("tokenizer not supported")
)

// ============================================================================
// Model Registry
// ============================================================================

// models speichert alle registrierten Modell-Konstruktoren.
var models = make(map[string]func(fs.Config) (Model, error))

// Register registriert einen Modell-Konstruktor fuer die gegebene Architektur.
func Register(name string, f func(fs.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

// New initialisiert eine neue Modell-Instanz mit der bereitgestellten
// Konfiguration basierend auf den Metadaten in der Modell-Datei.
func New(modelPath string, params ml.BackendParams) (Model, error) {
	b, err := ml.NewBackend(modelPath, params)
	if err != nil {
		return nil, err
	}

	m, err := modelForArch(b.Config())
	if err != nil {
		return nil, err
	}

	base := Base{b: b, config: m.Config()}
	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v.Elem()))
	return m, nil
}

// NewTextProcessor erstellt einen neuen TextProcessor aus einer Modell-Datei.
func NewTextProcessor(s string) (TextProcessor, error) {
	r, err := os.Open(s)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	m, err := modelForArch(meta.KV())
	if err != nil {
		return nil, err
	}

	tp, ok := m.(TextProcessor)
	if !ok {
		return nil, ErrUnsupportedTokenizer
	}
	return tp, nil
}

// modelForArch findet und erstellt ein Modell basierend auf der Architektur.
func modelForArch(c fs.Config) (Model, error) {
	arch := c.Architecture()
	if pooling.Type(c.Uint("pooling_type")) != pooling.TypeNone {
		arch = arch + "_embed"
	}

	f, ok := models[arch]
	if !ok {
		return nil, ErrUnsupportedModel
	}

	return f(c)
}

// ============================================================================
// Forward-Funktion
// ============================================================================

// Forward fuehrt den Vorwaerts-Pass des Modells aus.
// Validiert die Batch-Parameter und verwaltet den KV-Cache.
func Forward(ctx ml.Context, m Model, batch input.Batch) (ml.Tensor, error) {
	if len(batch.Positions) != len(batch.Sequences) {
		return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(batch.Positions), len(batch.Sequences))
	}

	if len(batch.Positions) < 1 {
		return nil, errors.New("batch size cannot be less than 1")
	}

	cache := m.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			return nil, err
		}
	}

	t, err := m.Forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)

	return t, nil
}
