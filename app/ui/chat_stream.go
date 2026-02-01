//go:build windows || darwin

// chat_stream.go - Chat Streaming Handler
// Enthält: chat(), browserState, reconstructBrowserState

package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/types/not"
	"github.com/ollama/ollama/app/ui/responses"
	"github.com/ollama/ollama/types/model"
)

// ErrNetworkOffline zeigt an, dass das Netzwerk nicht verfügbar ist
var ErrNetworkOffline = errors.New("network is offline")

// browserState holt den Browser-State aus dem Chat
func (s *Server) browserState(chat *store.Chat) (*responses.BrowserStateData, bool) {
	if len(chat.BrowserState) > 0 {
		var st responses.BrowserStateData
		if err := json.Unmarshal(chat.BrowserState, &st); err == nil {
			return &st, true
		}
	}
	return nil, false
}

// reconstructBrowserState rekonstruiert den Browser-State aus Messages (Legacy)
func reconstructBrowserState(messages []store.Message, defaultViewTokens int) *responses.BrowserStateData {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.ToolResult == nil {
			continue
		}
		var st responses.BrowserStateData
		if err := json.Unmarshal(*msg.ToolResult, &st); err == nil {
			if len(st.PageStack) > 0 || len(st.URLToPage) > 0 {
				if st.ViewTokens == 0 {
					st.ViewTokens = defaultViewTokens
				}
				return &st
			}
		}
	}
	return nil
}

// chat ist der Haupt-Streaming-Handler für Chat-Requests
func (s *Server) chat(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Content-Type", "text/jsonl")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("streaming not supported")
	}

	if r.Method != "POST" {
		return not.Found
	}

	cid := r.PathValue("id")
	createdChat := false

	// Neuen Chat erstellen wenn cid == "new"
	if cid == "new" {
		u, err := uuid.NewV7()
		if err != nil {
			return fmt.Errorf("failed to generate new chat id: %w", err)
		}
		cid = u.String()
		createdChat = true
	}

	var req responses.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "error unmarshalling body: %v\n", err)
		return fmt.Errorf("invalid request body: %w", err)
	}

	if req.Model == "" {
		return fmt.Errorf("empty model")
	}

	if req.Prompt == "" && !req.ForceUpdate {
		return fmt.Errorf("empty message")
	}

	if createdChat {
		json.NewEncoder(w).Encode(responses.ChatEvent{
			EventName: "chat_created",
			ChatID:    &cid,
		})
		flusher.Flush()
	}

	// Index für Message-Editing
	idx := -1
	if req.Index != nil {
		idx = *req.Index
	}

	// Chat mit Attachments laden
	chat, err := s.Store.ChatWithOptions(cid, true)
	if err != nil {
		if !errors.Is(err, not.Found) {
			return err
		}
		chat = store.NewChat(cid)
	}

	// User-Message hinzufügen (außer bei forceUpdate)
	if !req.ForceUpdate {
		chat, err = s.addUserMessage(chat, req, idx)
		if err != nil {
			return err
		}
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	_, cancelLoading := context.WithCancel(ctx)
	loading := false

	c, err := api.ClientFromEnvironment()
	if err != nil {
		cancelLoading()
		return err
	}

	// Model laden/pullen
	_, err = c.Show(ctx, &api.ShowRequest{Model: req.Model})
	if err != nil || req.ForceUpdate {
		chat.Messages = append(chat.Messages, store.NewMessage("assistant", "", &store.MessageOptions{Model: req.Model}))
		if err := s.Store.SetChat(*chat); err != nil {
			cancelLoading()
			return err
		}

		if err := s.pullModel(ctx, c, req.Model, w, flusher); err != nil {
			cancelLoading()
			return err
		}

		if req.ForceUpdate {
			json.NewEncoder(w).Encode(responses.ChatEvent{EventName: "done"})
			flusher.Flush()
			cancelLoading()
			return nil
		}
	}

	loading = true
	defer cancelLoading()

	// Model-Capabilities prüfen
	details, err := c.Show(ctx, &api.ShowRequest{Model: req.Model})
	if err != nil || details == nil {
		errorEvent := s.getError(err)
		json.NewEncoder(w).Encode(errorEvent)
		flusher.Flush()
		s.log().Error("failed to show model details", "error", err, "model", req.Model)
		return nil
	}

	think := slices.Contains(details.Capabilities, model.CapabilityThinking)
	thinkValue := s.determineThinkValue(req.Think, think)

	// Tools registrieren
	registry, browser := s.setupTools(chat, req)

	var thinkingTimeStart, thinkingTimeEnd *time.Time
	var pendingAssistantToolCalls []store.ToolCall

	// Haupt-Chat-Loop
	for passNum := 1; ; passNum++ {
		toolsExecuted, err := s.processChatPass(ctx, c, chat, req, registry, browser,
			thinkValue, &thinkingTimeStart, &thinkingTimeEnd, &pendingAssistantToolCalls,
			w, flusher, &loading)
		if err != nil {
			return err
		}

		if !toolsExecuted {
			break
		}
	}

	// Thinking-Zeit abschließen falls nötig
	if thinkingTimeStart != nil && thinkingTimeEnd == nil {
		now := time.Now()
		thinkingTimeEnd = &now
		if len(chat.Messages) > 0 && chat.Messages[len(chat.Messages)-1].Role == "assistant" {
			lastMsg := &chat.Messages[len(chat.Messages)-1]
			lastMsg.ThinkingTimeEnd = thinkingTimeEnd
			lastMsg.UpdatedAt = time.Now()
			s.Store.UpdateLastMessage(chat.ID, *lastMsg)
		}
	}

	json.NewEncoder(w).Encode(responses.ChatEvent{EventName: "done"})
	flusher.Flush()

	if len(chat.Messages) > 0 {
		chat.Messages[len(chat.Messages)-1].Stream = false
	}
	return s.Store.SetChat(*chat)
}
