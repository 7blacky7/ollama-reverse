//go:build windows || darwin

// chat_helpers.go - Chat-Hilfsfunktionen
// Enthält: addUserMessage, pullModel, determineThinkValue, setupTools, chatEventFromApiChatResponse

package ui

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui/responses"
)

// addUserMessage fügt eine User-Message zum Chat hinzu
func (s *Server) addUserMessage(chat *store.Chat, req responses.ChatRequest, idx int) (*store.Chat, error) {
	var messageOptions *store.MessageOptions
	if len(req.Attachments) > 0 {
		storeAttachments := make([]store.File, 0, len(req.Attachments))

		for _, att := range req.Attachments {
			if att.Data == "" {
				// Existierende Datei-Referenz behalten
				if idx >= 0 && idx < len(chat.Messages) {
					originalMessage := chat.Messages[idx]
					for _, originalFile := range originalMessage.Attachments {
						if originalFile.Filename == att.Filename {
							storeAttachments = append(storeAttachments, originalFile)
							break
						}
					}
				}
			} else {
				// Neue Datei dekodieren
				data, err := base64.StdEncoding.DecodeString(att.Data)
				if err != nil {
					s.log().Error("failed to decode attachment data", "error", err, "filename", att.Filename)
					continue
				}

				storeAttachments = append(storeAttachments, store.File{
					Filename: att.Filename,
					Data:     data,
				})
			}
		}

		messageOptions = &store.MessageOptions{
			Attachments: storeAttachments,
		}
	}

	userMsg := store.NewMessage("user", req.Prompt, messageOptions)

	if idx >= 0 && idx < len(chat.Messages) {
		chat.Messages = chat.Messages[:idx]
		chat.Messages = append(chat.Messages, userMsg)
	} else {
		chat.Messages = append(chat.Messages, userMsg)
	}

	if err := s.Store.SetChat(*chat); err != nil {
		return nil, err
	}
	return chat, nil
}

// pullModel lädt ein Model und sendet Progress-Events
func (s *Server) pullModel(ctx context.Context, c *api.Client, model string, w http.ResponseWriter, flusher http.Flusher) error {
	var largestDigest string
	var largestTotal int64

	err := c.Pull(ctx, &api.PullRequest{Model: model}, func(progress api.ProgressResponse) error {
		if progress.Digest != "" && progress.Total > largestTotal {
			largestDigest = progress.Digest
			largestTotal = progress.Total
		}

		if progress.Digest != "" && progress.Digest == largestDigest {
			progressEvent := responses.DownloadEvent{
				EventName: string(EventDownload),
				Total:     progress.Total,
				Completed: progress.Completed,
				Done:      false,
			}

			if err := json.NewEncoder(w).Encode(progressEvent); err != nil {
				return err
			}
			flusher.Flush()
		}
		return nil
	})

	if err != nil {
		s.log().Error("model download error", "error", err, "model", model)
		errorEvent := s.getError(err)
		json.NewEncoder(w).Encode(errorEvent)
		flusher.Flush()
		return fmt.Errorf("failed to download model: %w", err)
	}

	if err := json.NewEncoder(w).Encode(responses.DownloadEvent{
		EventName: string(EventDownload),
		Completed: largestTotal,
		Total:     largestTotal,
		Done:      true,
	}); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// determineThinkValue bestimmt den Think-Wert für den Request
func (s *Server) determineThinkValue(reqThink *bool, modelThink bool) any {
	if reqThink != nil {
		return reqThink
	}
	return modelThink
}

// setupTools registriert die verfügbaren Tools
func (s *Server) setupTools(chat *store.Chat, req responses.ChatRequest) (*tools.Registry, *tools.Browser) {
	hasAttachments := false
	if len(chat.Messages) > 0 {
		lastMsg := chat.Messages[len(chat.Messages)-1]
		if lastMsg.Role == "user" && len(lastMsg.Attachments) > 0 {
			hasAttachments = true
		}
	}

	registry := tools.NewRegistry()
	var browser *tools.Browser

	if !hasAttachments {
		WebSearchEnabled := req.WebSearch != nil && *req.WebSearch

		if WebSearchEnabled {
			if supportsBrowserTools(req.Model) {
				browserState, ok := s.browserState(chat)
				if !ok {
					browserState = reconstructBrowserState(chat.Messages, tools.DefaultViewTokens)
				}
				browser = tools.NewBrowser(browserState)
				registry.Register(tools.NewBrowserSearch(browser))
				registry.Register(tools.NewBrowserOpen(browser))
				registry.Register(tools.NewBrowserFind(browser))
			} else if supportsWebSearchTools(req.Model) {
				registry.Register(&tools.WebSearch{})
				registry.Register(&tools.WebFetch{})
			}
		}
	}

	return registry, browser
}

// chatEventFromApiChatResponse konvertiert API-Response zu ChatEvent
func chatEventFromApiChatResponse(res api.ChatResponse, thinkingTimeStart *time.Time, thinkingTimeEnd *time.Time) responses.ChatEvent {
	if len(res.Message.ToolCalls) > 0 {
		storeToolCalls := make([]store.ToolCall, len(res.Message.ToolCalls))
		for i, tc := range res.Message.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Function.Arguments)
			storeToolCalls[i] = store.ToolCall{
				Type: "function",
				Function: store.ToolFunction{
					Name:      tc.Function.Name,
					Arguments: string(argsJSON),
				},
			}
		}

		var content *string
		if res.Message.Content != "" {
			content = &res.Message.Content
		}
		var thinking *string
		if res.Message.Thinking != "" {
			thinking = &res.Message.Thinking
		}

		return responses.ChatEvent{
			EventName:         "assistant_with_tools",
			Content:           content,
			Thinking:          thinking,
			ToolCalls:         storeToolCalls,
			ThinkingTimeStart: thinkingTimeStart,
			ThinkingTimeEnd:   thinkingTimeEnd,
		}
	}

	var content *string
	if res.Message.Content != "" {
		content = &res.Message.Content
	}
	var thinking *string
	if res.Message.Thinking != "" {
		thinking = &res.Message.Thinking
	}

	return responses.ChatEvent{
		EventName:         "chat",
		Content:           content,
		Thinking:          thinking,
		ThinkingTimeStart: thinkingTimeStart,
		ThinkingTimeEnd:   thinkingTimeEnd,
	}
}
