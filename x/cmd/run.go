// run.go - Haupt-Agenten-Chat-Loop
//
// Dieses Modul enthält:
// - RunOptions: Konfiguration für Chat-Session
// - Chat: Agenten-Chat-Loop mit Tool-Support

package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/x/agent"
	"github.com/ollama/ollama/x/tools"
)

// RunOptions enthält Optionen für eine interaktive Agent-Session.
type RunOptions struct {
	Model        string
	Messages     []api.Message
	WordWrap     bool
	Format       string
	System       string
	Options      map[string]any
	KeepAlive    *api.Duration
	Think        *api.ThinkValue
	HideThinking bool
	Verbose      bool

	// Agent-Felder (extern verwaltet für Session-Persistenz)
	Tools    *tools.Registry
	Approval *agent.ApprovalManager

	// YoloMode überspringt alle Tool-Approval-Prompts
	YoloMode bool
}

// Chat führt einen Agenten-Chat-Loop mit Tool-Support aus.
// Dies ist die experimentelle Version von Chat mit Tool-Calling-Unterstützung.
func Chat(ctx context.Context, opts RunOptions) (*api.Message, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	// Tools-Registry und Approval aus opts (vom Aufrufer für Session-Persistenz verwaltet)
	toolRegistry := opts.Tools
	approval := opts.Approval
	if approval == nil {
		approval = agent.NewApprovalManager()
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	cancelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var fullResponse strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false
	var pendingToolCalls []api.ToolCall
	var consecutiveErrors int // Zählt konsekutive 500-Fehler für Retry-Limit
	var latest api.ChatResponse

	role := "assistant"
	messages := opts.Messages

	fn := func(response api.ChatResponse) error {
		if response.Message.Content != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		latest = response
		role = response.Message.Role
		if response.Message.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(false))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Message.Thinking)
			displayResponse(response.Message.Thinking, opts.WordWrap, state)
		}

		content := response.Message.Content
		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.Message.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(false))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}

		fullResponse.WriteString(content)

		if response.Message.ToolCalls != nil {
			toolCalls := response.Message.ToolCalls
			if len(toolCalls) > 0 {
				if toolRegistry != nil {
					// Tool-Calls für Ausführung nach Response-Ende speichern
					pendingToolCalls = append(pendingToolCalls, toolCalls...)
				} else {
					// Keine Tools-Registry, nur Tool-Calls anzeigen
					fmt.Print(renderToolCalls(toolCalls, false))
				}
			}
		}

		displayResponse(content, opts.WordWrap, state)

		return nil
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	// Agenten-Loop: fortsetzen bis keine Tool-Calls mehr
	for {
		req := &api.ChatRequest{
			Model:    opts.Model,
			Messages: messages,
			Format:   json.RawMessage(opts.Format),
			Options:  opts.Options,
			Think:    opts.Think,
		}

		// Tools hinzufügen
		if toolRegistry != nil {
			apiTools := toolRegistry.Tools()
			if len(apiTools) > 0 {
				req.Tools = apiTools
			}
		}

		if opts.KeepAlive != nil {
			req.KeepAlive = opts.KeepAlive
		}

		if err := client.Chat(cancelCtx, req, fn); err != nil {
			if errors.Is(err, context.Canceled) {
				return nil, nil
			}

			// Bei 401 Unauthorized - Benutzer zum Einloggen auffordern
			var authErr api.AuthorizationError
			if errors.As(err, &authErr) {
				p.StopAndClear()
				fmt.Fprintf(os.Stderr, "\033[1mauth required:\033[0m cloud model requires authentication\n")
				result, promptErr := agent.PromptYesNo("Sign in to Ollama?")
				if promptErr == nil && result {
					if signinErr := waitForOllamaSignin(ctx); signinErr == nil {
						// Chat-Request erneut versuchen
						fmt.Fprintf(os.Stderr, "\033[90mretrying...\033[0m\n")
						continue
					}
				}
				return nil, fmt.Errorf("authentication required - run 'ollama signin' to authenticate")
			}

			// Bei 500-Fehlern (oft Tool-Parsing-Fehler) - Modell informieren
			var statusErr api.StatusError
			if errors.As(err, &statusErr) && statusErr.StatusCode >= 500 {
				consecutiveErrors++
				p.StopAndClear()

				if consecutiveErrors >= 3 {
					fmt.Fprintf(os.Stderr, "\033[1merror:\033[0m too many consecutive errors, giving up\n")
					return nil, fmt.Errorf("too many consecutive server errors: %s", statusErr.ErrorMessage)
				}

				fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m server error (attempt %d/3): %s\n", consecutiveErrors, statusErr.ErrorMessage)

				// Modell-Response und Fehler inkludieren damit es lernen kann
				assistantContent := fullResponse.String()
				if assistantContent == "" {
					assistantContent = "(empty response)"
				}
				errorMsg := fmt.Sprintf("Your previous response caused an error: %s\n\nYour response was:\n%s\n\nPlease try again with a valid response.", statusErr.ErrorMessage, assistantContent)
				messages = append(messages,
					api.Message{Role: "user", Content: errorMsg},
				)

				// State zurücksetzen und erneut versuchen
				fullResponse.Reset()
				thinkingContent.Reset()
				thinkTagOpened = false
				thinkTagClosed = false
				pendingToolCalls = nil
				state = &displayResponseState{}
				p = progress.NewProgress(os.Stderr)
				spinner = progress.NewSpinner("")
				p.Add("", spinner)
				continue
			}

			if strings.Contains(err.Error(), "upstream error") {
				p.StopAndClear()
				fmt.Println("An error occurred while processing your message. Please try again.")
				fmt.Println()
				return nil, nil
			}
			return nil, err
		}

		// Konsekutive-Fehler-Zähler bei Erfolg zurücksetzen
		consecutiveErrors = 0

		// Wenn keine Tool-Calls, sind wir fertig
		if len(pendingToolCalls) == 0 || toolRegistry == nil {
			break
		}

		// Tool-Calls ausführen und Konversation fortsetzen
		fmt.Fprintf(os.Stderr, "\n")

		// Assistant-Tool-Call-Message zur History hinzufügen
		assistantMsg := api.Message{
			Role:      "assistant",
			Content:   fullResponse.String(),
			Thinking:  thinkingContent.String(),
			ToolCalls: pendingToolCalls,
		}
		messages = append(messages, assistantMsg)

		// Jeden Tool-Call ausführen und Ergebnisse sammeln
		var toolResults []api.Message
		for _, call := range pendingToolCalls {
			result := executeToolCall(ctx, call, toolRegistry, approval, opts)
			toolResults = append(toolResults, result)
		}

		// Tool-Ergebnisse zur Message-History hinzufügen
		messages = append(messages, toolResults...)

		fmt.Fprintf(os.Stderr, "\n")

		// State für nächste Iteration zurücksetzen
		fullResponse.Reset()
		thinkingContent.Reset()
		thinkTagOpened = false
		thinkTagClosed = false
		pendingToolCalls = nil
		state = &displayResponseState{}

		// Neuen Progress-Spinner für nächsten API-Call starten
		p = progress.NewProgress(os.Stderr)
		spinner = progress.NewSpinner("")
		p.Add("", spinner)
	}

	if len(opts.Messages) > 0 {
		fmt.Println()
		fmt.Println()
	}

	if opts.Verbose {
		latest.Summary()
	}

	return &api.Message{Role: role, Thinking: thinkingContent.String(), Content: fullResponse.String()}, nil
}
