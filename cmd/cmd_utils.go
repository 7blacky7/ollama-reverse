// cmd_utils.go - Gemeinsame Hilfsfunktionen
// Hauptfunktionen: loadOrUnloadModel, checkServerHeartbeat, display*
package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
)

// runOptions - Optionen fuer Model-Ausfuehrung
type runOptions struct {
	Model        string
	ParentModel  string
	Prompt       string
	Messages     []api.Message
	WordWrap     bool
	Format       string
	System       string
	Images       []api.ImageData
	Options      map[string]any
	MultiModal   bool
	KeepAlive    *api.Duration
	Think        *api.ThinkValue
	HideThinking bool
	ShowConnect  bool
}

// Copy - Erstellt eine Kopie der runOptions
func (r runOptions) Copy() runOptions {
	var messages []api.Message
	if r.Messages != nil {
		messages = make([]api.Message, len(r.Messages))
		copy(messages, r.Messages)
	}

	var images []api.ImageData
	if r.Images != nil {
		images = make([]api.ImageData, len(r.Images))
		copy(images, r.Images)
	}

	var opts map[string]any
	if r.Options != nil {
		opts = make(map[string]any, len(r.Options))
		for k, v := range r.Options {
			opts[k] = v
		}
	}

	var think *api.ThinkValue
	if r.Think != nil {
		cThink := *r.Think
		think = &cThink
	}

	return runOptions{
		Model:        r.Model,
		ParentModel:  r.ParentModel,
		Prompt:       r.Prompt,
		Messages:     messages,
		WordWrap:     r.WordWrap,
		Format:       r.Format,
		System:       r.System,
		Images:       images,
		Options:      opts,
		MultiModal:   r.MultiModal,
		KeepAlive:    r.KeepAlive,
		Think:        think,
		HideThinking: r.HideThinking,
		ShowConnect:  r.ShowConnect,
	}
}

type displayResponseState struct {
	lineLength int
	wordBuffer string
}

// loadOrUnloadModel - Laedt oder entlaedt ein Modell
func loadOrUnloadModel(cmd *cobra.Command, opts *runOptions) error {
	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if info, err := client.Show(cmd.Context(), &api.ShowRequest{Model: opts.Model}); err != nil {
		return err
	} else if info.RemoteHost != "" {
		if opts.ShowConnect {
			p.StopAndClear()
			if strings.HasPrefix(info.RemoteHost, "https://ollama.com") {
				fmt.Fprintf(os.Stderr, "Connecting to '%s' on 'ollama.com'\n", info.RemoteModel)
			} else {
				fmt.Fprintf(os.Stderr, "Connecting to '%s' on '%s'\n", info.RemoteModel, info.RemoteHost)
			}
		}
		return nil
	}

	req := &api.GenerateRequest{
		Model:     opts.Model,
		KeepAlive: opts.KeepAlive,
		Think:     opts.Think,
	}

	return client.Generate(cmd.Context(), req, func(r api.GenerateResponse) error {
		return nil
	})
}

// checkServerHeartbeat - Prueft ob der Server erreichbar ist
func checkServerHeartbeat(cmd *cobra.Command, _ []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(cmd.Context()); err != nil {
		if !(strings.Contains(err.Error(), " refused") || strings.Contains(err.Error(), "could not connect")) {
			return err
		}
		if err := startApp(cmd.Context(), client); err != nil {
			return fmt.Errorf("ollama server not responding - %w", err)
		}
	}
	return nil
}

// ensureThinkingSupport - Warnt wenn Modell kein Thinking unterstuetzt
func ensureThinkingSupport(ctx context.Context, client *api.Client, name string) {
	if name == "" {
		return
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		return
	}
	if slices.Contains(resp.Capabilities, model.CapabilityThinking) {
		return
	}
	fmt.Fprintf(os.Stderr, "warning: model %q does not support thinking output\n", name)
}

// inferThinkingOption - Ermittelt automatisch Thinking-Option
func inferThinkingOption(caps *[]model.Capability, runOpts *runOptions, explicitlySetByUser bool) (*api.ThinkValue, error) {
	if explicitlySetByUser {
		return runOpts.Think, nil
	}

	if caps == nil {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			return nil, err
		}
		ret, err := client.Show(context.Background(), &api.ShowRequest{
			Model: runOpts.Model,
		})
		if err != nil {
			return nil, err
		}
		caps = &ret.Capabilities
	}

	for _, cap := range *caps {
		if cap == model.CapabilityThinking {
			return &api.ThinkValue{Value: true}, nil
		}
	}

	return nil, nil
}

// readStdinContent - Liest Inhalt von stdin
func readStdinContent() (string, error) {
	in, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", err
	}
	return string(in), nil
}

// getModelInfo - Holt Model-Informationen, pullt bei Bedarf
func getModelInfo(cmd *cobra.Command, client *api.Client, name string) (*api.ShowResponse, error) {
	showReq := &api.ShowRequest{Name: name}
	info, err := client.Show(cmd.Context(), showReq)
	var se api.StatusError
	if errors.As(err, &se) && se.StatusCode == http.StatusNotFound {
		if err := PullHandler(cmd, []string{name}); err != nil {
			return nil, err
		}
		return client.Show(cmd.Context(), &api.ShowRequest{Name: name})
	}
	return info, err
}

// handleEmbeddingModel - Behandelt Embedding-Modelle
func handleEmbeddingModel(cmd *cobra.Command, name string, opts runOptions) error {
	if opts.Prompt == "" {
		return errors.New("embedding models require input text. Usage: ollama run " + name + " \"your text here\"")
	}

	var truncate *bool
	if truncateFlag, err := cmd.Flags().GetBool("truncate"); err == nil && cmd.Flags().Changed("truncate") {
		truncate = &truncateFlag
	}

	dimensions, err := cmd.Flags().GetInt("dimensions")
	if err != nil {
		return err
	}

	return generateEmbedding(cmd, name, opts.Prompt, opts.KeepAlive, truncate, dimensions)
}

// generateEmbedding - Generiert Embeddings fuer Text
func generateEmbedding(cmd *cobra.Command, modelName, input string, keepAlive *api.Duration, truncate *bool, dimensions int) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := &api.EmbedRequest{
		Model: modelName,
		Input: input,
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}
	if truncate != nil {
		req.Truncate = truncate
	}
	if dimensions > 0 {
		req.Dimensions = dimensions
	}

	resp, err := client.Embed(cmd.Context(), req)
	if err != nil {
		return err
	}

	if len(resp.Embeddings) == 0 {
		return errors.New("no embeddings returned")
	}

	output, err := json.Marshal(resp.Embeddings[0])
	if err != nil {
		return err
	}
	fmt.Println(string(output))

	return nil
}
