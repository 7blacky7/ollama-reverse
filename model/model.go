// Package model - Model-Interface und Initialisierung
//
// Dieses Paket definiert das Model-Interface und stellt Funktionen
// zur Initialisierung und Verwaltung von ML-Modellen bereit.
//
// Hauptkomponenten:
// - Model: Interface für alle Modell-Architekturen
// - Base: Basis-Implementierung für gemeinsame Funktionalität
// - New: Erstellt neue Model-Instanzen
// - Register: Registriert Modell-Konstruktoren
// - Forward: Führt Vorwärts-Pass durch

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
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model/input"
)

// Fehler-Definitionen
var (
	ErrNoVisionModel        = errors.New("this model is missing data required for image input")
	ErrUnsupportedModel     = errors.New("model not supported")
	ErrUnsupportedTokenizer = errors.New("tokenizer not supported")
)

// Model definiert das Interface für spezifische Modell-Architekturen
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
}

// Validator ist ein optionales Interface für Post-Load-Validierung
type Validator interface {
	Validate() error
}

// MultimodalProcessor muss von multimodalen Modellen implementiert werden
type MultimodalProcessor interface {
	// EncodeMultimodal verarbeitet eine einzelne Eingabe (z.B. Bild)
	// und generiert einen Output (typischerweise ein Embedding)
	EncodeMultimodal(ml.Context, []byte) ([]input.Multimodal, error)

	// PostTokenize wird nach der Tokenisierung aufgerufen um
	// multimodale Elemente korrekt anzuordnen
	PostTokenize([]*input.Input) ([]*input.Input, error)
}

// Base implementiert gemeinsame Felder und Methoden für alle Modelle
type Base struct {
	b ml.Backend
	config
}

// config enthält die Modell-Konfiguration
type config struct {
	Cache kvcache.Cache
}

// Backend gibt das Backend zurück, das das Modell ausführt
func (m *Base) Backend() ml.Backend {
	return m.b
}

// Config gibt die Modell-Konfiguration zurück
func (m *Base) Config() config {
	return m.config
}

// models speichert registrierte Modell-Konstruktoren
var models = make(map[string]func(fs.Config) (Model, error))

// Register registriert einen Modell-Konstruktor für eine Architektur
func Register(name string, f func(fs.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

// New initialisiert eine neue Model-Instanz basierend auf den Metadaten
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

	if validator, ok := m.(Validator); ok {
		if err := validator.Validate(); err != nil {
			return nil, err
		}
	}

	return m, nil
}

// NewTextProcessor erstellt einen TextProcessor aus einer Modell-Datei
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

// modelForArch erstellt ein Model basierend auf der Architektur
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

// Forward führt einen Vorwärts-Pass durch das Modell aus
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
