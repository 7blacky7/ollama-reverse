// Package server - Generate Types und Helper-Funktionen
// Beinhaltet: Konstanten, Fehlerdefinitionen, gemeinsame Utility-Funktionen
// fuer die Generate-Handler Logik
package server

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

const signinURLStr = "https://ollama.com/connect?name=%s&key=%s"

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

// shouldUseHarmony prueft ob Harmony-Parser verwendet werden soll
func shouldUseHarmony(model *Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, model.Config.ModelFamily) {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

// experimentEnabled prueft ob ein Experiment aktiviert ist
func experimentEnabled(name string) bool {
	return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")

// modelOptions merged Model-Optionen mit Request-Optionen
func modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	opts := api.DefaultOptions()
	if err := opts.FromMap(model.Options); err != nil {
		return api.Options{}, err
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	return opts, nil
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, and consolidated options if successful and error otherwise.
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	m, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(m.ProjectorPaths) > 0 {
		return nil, nil, nil, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	if err := m.CheckCapabilities(caps...); err != nil {
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	opts, err := modelOptions(m, requestOpts)
	if err != nil {
		return nil, nil, nil, err
	}

	// This model is much more capable with a larger context, so set that
	// unless it would penalize performance too much
	if !s.lowVRAM && slices.Contains([]string{
		"gptoss", "gpt-oss",
		"qwen3vl", "qwen3vlmoe",
	}, m.Config.ModelFamily) {
		opts.NumCtx = max(opts.NumCtx, 8192)
	}

	runnerCh, errCh := s.sched.GetRunner(ctx, m, opts, keepAlive)
	var runner *runnerRef
	select {
	case runner = <-runnerCh:
	case err = <-errCh:
		return nil, nil, nil, err
	}

	return runner.llama, m, &opts, nil
}

// signinURL generiert die Sign-In URL
func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf(signinURLStr, url.PathEscape(h), encKey), nil
}

// toolCallId generiert eine zufaellige Tool-Call ID
func toolCallId() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

// handleScheduleError behandelt Scheduler-Fehler
func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.Canceled):
		c.JSON(499, gin.H{"error": "request canceled"})
	case errors.Is(err, ErrMaxQueue):
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

// filterThinkTags filtert Think-Tags aus Nachrichten
func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}

		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				// TODO(drifkin): this is from before we added proper thinking support.
				// However, even if thinking is not enabled (and therefore we shouldn't
				// change the user output), we should probably perform this filtering
				// for all thinking models (not just qwen3 & deepseek-r1) since it tends
				// to save tokens and improve quality.
				thinkingState := &thinking.Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}
