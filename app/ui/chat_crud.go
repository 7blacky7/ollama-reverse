//go:build windows || darwin

// chat_crud.go - Chat CRUD-Operationen
// Enthält: createChat, listChats, getChat, renameChat, deleteChat

package ui

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/types/not"
	"github.com/ollama/ollama/app/ui/responses"
)

// createChat erstellt einen neuen Chat
func (s *Server) createChat(w http.ResponseWriter, r *http.Request) error {
	if err := WaitForServer(r.Context(), 10*time.Second); err != nil {
		return err
	}

	id, err := uuid.NewV7()
	if err != nil {
		return fmt.Errorf("failed to generate chat ID: %w", err)
	}

	json.NewEncoder(w).Encode(map[string]string{"id": id.String()})
	return nil
}

// listChats listet alle Chats auf
func (s *Server) listChats(w http.ResponseWriter, r *http.Request) error {
	chats, _ := s.Store.Chats()

	chatInfos := make([]responses.ChatInfo, len(chats))
	for i, chat := range chats {
		chatInfos[i] = chatInfoFromChat(chat)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responses.ChatsResponse{ChatInfos: chatInfos})
	return nil
}

// getChat holt einen Chat mit allen Nachrichten
func (s *Server) getChat(w http.ResponseWriter, r *http.Request) error {
	cid := r.PathValue("id")

	if cid == "" {
		return fmt.Errorf("chat ID is required")
	}

	chat, err := s.Store.Chat(cid)
	if err != nil {
		// Leeren Chat zurückgeben wenn nicht gefunden
		data := responses.ChatResponse{
			Chat: store.Chat{},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(data)
		return nil //nolint:nilerr
	}

	// tool_name auf tool-Messages füllen (aus vorherigen tool_calls)
	if chat != nil && len(chat.Messages) > 0 {
		for i := range chat.Messages {
			if chat.Messages[i].Role == "tool" && chat.Messages[i].ToolName == "" && chat.Messages[i].ToolResult != nil {
				for j := i - 1; j >= 0; j-- {
					if chat.Messages[j].Role == "assistant" && len(chat.Messages[j].ToolCalls) > 0 {
						last := chat.Messages[j].ToolCalls[len(chat.Messages[j].ToolCalls)-1]
						if last.Function.Name != "" {
							chat.Messages[i].ToolName = last.Function.Name
						}
						break
					}
				}
			}
		}
	}

	browserState, ok := s.browserState(chat)
	if !ok {
		browserState = reconstructBrowserState(chat.Messages, tools.DefaultViewTokens)
	}
	// Text und Lines aus allen Pages entfernen (nicht für Rendering benötigt)
	if browserState != nil {
		for _, page := range browserState.URLToPage {
			page.Lines = nil
			page.Text = ""
		}

		if cleanedState, err := json.Marshal(browserState); err == nil {
			chat.BrowserState = json.RawMessage(cleanedState)
		}
	}

	data := responses.ChatResponse{
		Chat: *chat,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
	return nil
}

// renameChat benennt einen Chat um
func (s *Server) renameChat(w http.ResponseWriter, r *http.Request) error {
	cid := r.PathValue("id")
	if cid == "" {
		return fmt.Errorf("chat ID is required")
	}

	var req struct {
		Title string `json:"title"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}

	// Chat ohne Attachments laden (nur Titel wird geändert)
	chat, err := s.Store.ChatWithOptions(cid, false)
	if err != nil {
		return fmt.Errorf("chat not found: %w", err)
	}

	chat.Title = req.Title
	if err := s.Store.SetChat(*chat); err != nil {
		return fmt.Errorf("failed to update chat: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chatInfoFromChat(*chat))
	return nil
}

// deleteChat löscht einen Chat
func (s *Server) deleteChat(w http.ResponseWriter, r *http.Request) error {
	cid := r.PathValue("id")
	if cid == "" {
		return fmt.Errorf("chat ID is required")
	}

	// Prüfen ob Chat existiert
	_, err := s.Store.ChatWithOptions(cid, false)
	if err != nil {
		if errors.Is(err, not.Found) {
			w.WriteHeader(http.StatusNotFound)
			return fmt.Errorf("chat not found")
		}
		return fmt.Errorf("failed to get chat: %w", err)
	}

	if err := s.Store.DeleteChat(cid); err != nil {
		return fmt.Errorf("failed to delete chat: %w", err)
	}

	w.WriteHeader(http.StatusOK)
	return nil
}

// chatInfoFromChat extrahiert ChatInfo aus einem Chat
func chatInfoFromChat(chat store.Chat) responses.ChatInfo {
	userExcerpt := ""
	var updatedAt time.Time

	for _, msg := range chat.Messages {
		if msg.Role == "user" && userExcerpt == "" {
			userExcerpt = msg.Content
		}
		if msg.UpdatedAt.After(updatedAt) {
			updatedAt = msg.UpdatedAt
		}
	}

	return responses.ChatInfo{
		ID:          chat.ID,
		Title:       chat.Title,
		UserExcerpt: userExcerpt,
		CreatedAt:   chat.CreatedAt,
		UpdatedAt:   updatedAt,
	}
}
