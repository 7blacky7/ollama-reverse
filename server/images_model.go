// Package server - Model-Strukturen und Capabilities
//
// Diese Datei enthält:
// - Model struct Definition
// - Capabilities-Verwaltung (Capabilities, CheckCapabilities)
// - Model String-Serialisierung
package server

import (
	"errors"
	"fmt"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// Fehler für fehlende Capabilities
var (
	errCapabilities         = errors.New("does not support")
	errCapabilityCompletion = errors.New("completion")
	errCapabilityTools      = errors.New("tools")
	errCapabilityInsert     = errors.New("insert")
	errCapabilityVision     = errors.New("vision")
	errCapabilityEmbedding  = errors.New("embedding")
	errCapabilityThinking   = errors.New("thinking")
	errCapabilityImage      = errors.New("image generation")
)

// Model repräsentiert ein geladenes Ollama-Modell mit allen Metadaten
type Model struct {
	Name           string `json:"name"`
	Config         model.ConfigV2
	ShortName      string
	ModelPath      string
	ParentModel    string
	AdapterPaths   []string
	ProjectorPaths []string
	System         string
	License        []string
	Digest         string
	Options        map[string]any
	Messages       []api.Message

	Template *template.Template
}

// Capabilities ermittelt die verfügbaren Fähigkeiten des Modells
func (m *Model) Capabilities() []model.Capability {
	capabilities := []model.Capability{}

	if m.ModelPath != "" {
		f, err := gguf.Open(m.ModelPath)
		if err == nil {
			defer f.Close()

			if f.KeyValue("pooling_type").Valid() {
				capabilities = append(capabilities, model.CapabilityEmbedding)
			} else {
				// Kein Embedding definiert -> Completion wird angenommen
				capabilities = append(capabilities, model.CapabilityCompletion)
			}
			if f.KeyValue("vision.block_count").Valid() {
				capabilities = append(capabilities, model.CapabilityVision)
			}
		} else {
			slog.Error("couldn't open model file", "error", err)
		}
	} else if len(m.Config.Capabilities) > 0 {
		for _, c := range m.Config.Capabilities {
			capabilities = append(capabilities, model.Capability(c))
		}
	} else {
		slog.Warn("unknown capabilities for model", "model", m.Name)
	}

	if m.Template == nil {
		return capabilities
	}

	builtinParser := parsers.ParserForName(m.Config.Parser)
	// Tools-Capability prüfen
	v, err := m.Template.Vars()
	if err != nil {
		slog.Warn("model template contains errors", "error", err)
	}
	if slices.Contains(v, "tools") || (builtinParser != nil && builtinParser.HasToolSupport()) {
		capabilities = append(capabilities, model.CapabilityTools)
	}

	// Insert-Capability prüfen
	if slices.Contains(v, "suffix") {
		capabilities = append(capabilities, model.CapabilityInsert)
	}

	// Vision-Capability bei Projector-basierten Modellen
	if len(m.ProjectorPaths) > 0 {
		capabilities = append(capabilities, model.CapabilityVision)
	}

	// Thinking-Check überspringen falls bereits gesetzt
	if slices.Contains(capabilities, "thinking") {
		return capabilities
	}

	// Thinking-Capability prüfen
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	hasTags := openingTag != "" && closingTag != ""
	isGptoss := slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily)
	if hasTags || isGptoss || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

// CheckCapabilities prüft ob das Modell die gewünschten Capabilities hat
// Gibt einen Fehler zurück mit Beschreibung der fehlenden Capabilities
func (m *Model) CheckCapabilities(want ...model.Capability) error {
	available := m.Capabilities()
	var errs []error

	// Mapping von Capability zu Fehler
	capToErr := map[model.Capability]error{
		model.CapabilityCompletion: errCapabilityCompletion,
		model.CapabilityTools:      errCapabilityTools,
		model.CapabilityInsert:     errCapabilityInsert,
		model.CapabilityVision:     errCapabilityVision,
		model.CapabilityEmbedding:  errCapabilityEmbedding,
		model.CapabilityThinking:   errCapabilityThinking,
		model.CapabilityImage:      errCapabilityImage,
	}

	for _, cap := range want {
		err, ok := capToErr[cap]
		if !ok {
			slog.Error("unknown capability", "capability", cap)
			return fmt.Errorf("unknown capability: %s", cap)
		}

		if !slices.Contains(available, cap) {
			errs = append(errs, err)
		}
	}

	var err error
	if len(errs) > 0 {
		err = fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))
	}

	if slices.Contains(errs, errCapabilityThinking) {
		if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
			// Hinweis auf Update für Thinking-Support
			return fmt.Errorf("%w. Pull the model again to get the latest version with full thinking support", err)
		}
	}

	return err
}

// String serialisiert das Model zu einem Modelfile-Format
func (m *Model) String() string {
	var modelfile parser.Modelfile

	modelfile.Commands = append(modelfile.Commands, parser.Command{
		Name: "model",
		Args: m.ModelPath,
	})

	for _, adapter := range m.AdapterPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "adapter",
			Args: adapter,
		})
	}

	for _, projector := range m.ProjectorPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "model",
			Args: projector,
		})
	}

	if m.Template != nil {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "template",
			Args: m.Template.String(),
		})
	}

	if m.System != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "system",
			Args: m.System,
		})
	}

	if m.Config.Renderer != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "renderer",
			Args: m.Config.Renderer,
		})
	}

	if m.Config.Parser != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "parser",
			Args: m.Config.Parser,
		})
	}

	for k, v := range m.Options {
		switch v := v.(type) {
		case []any:
			for _, s := range v {
				modelfile.Commands = append(modelfile.Commands, parser.Command{
					Name: k,
					Args: fmt.Sprintf("%v", s),
				})
			}
		default:
			modelfile.Commands = append(modelfile.Commands, parser.Command{
				Name: k,
				Args: fmt.Sprintf("%v", v),
			})
		}
	}

	for _, license := range m.License {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "license",
			Args: license,
		})
	}

	for _, msg := range m.Messages {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "message",
			Args: fmt.Sprintf("%s: %s", msg.Role, msg.Content),
		})
	}

	return modelfile.String()
}
