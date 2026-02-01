//go:build windows || darwin

// chat_process.go - Chat-Verarbeitung und Event-Handling
// Enthält: processChatPass, handleChatEvent, handleThinkingEvent

package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui/responses"
)

// processChatPass führt einen Chat-Durchlauf durch
func (s *Server) processChatPass(
	ctx context.Context,
	c *api.Client,
	chat *store.Chat,
	req responses.ChatRequest,
	registry *tools.Registry,
	browser *tools.Browser,
	thinkValue any,
	thinkingTimeStart, thinkingTimeEnd **time.Time,
	pendingAssistantToolCalls *[]store.ToolCall,
	w http.ResponseWriter,
	flusher http.Flusher,
	loading *bool,
) (bool, error) {
	var toolsExecuted bool
	availableTools := registry.AvailableTools()

	// Request-Chat mit pending tool_calls vorbereiten
	reqChat := chat
	if len(*pendingAssistantToolCalls) > 0 {
		if len(chat.Messages) == 0 || chat.Messages[len(chat.Messages)-1].Role != "assistant" {
			temp := *chat
			synth := store.NewMessage("assistant", "", &store.MessageOptions{Model: req.Model, ToolCalls: *pendingAssistantToolCalls})
			insertIdx := len(temp.Messages) - 1
			for insertIdx >= 0 && temp.Messages[insertIdx].Role == "tool" {
				insertIdx--
			}
			if insertIdx < 0 {
				temp.Messages = append([]store.Message{synth}, temp.Messages...)
			} else {
				tmp := make([]store.Message, 0, len(temp.Messages)+1)
				tmp = append(tmp, temp.Messages[:insertIdx+1]...)
				tmp = append(tmp, synth)
				tmp = append(tmp, temp.Messages[insertIdx+1:]...)
				temp.Messages = tmp
			}
			reqChat = &temp
		}
	}

	chatReq, err := s.buildChatRequest(reqChat, req.Model, thinkValue, availableTools)
	if err != nil {
		return false, err
	}

	err = c.Chat(ctx, chatReq, func(res api.ChatResponse) error {
		if *loading {
			*loading = false
		}

		// Thinking-Timer starten
		if res.Message.Thinking != "" && (*thinkingTimeStart == nil || *thinkingTimeEnd != nil) {
			now := time.Now()
			*thinkingTimeStart = &now
			*thinkingTimeEnd = nil
		}

		if res.Message.Content == "" && res.Message.Thinking == "" && len(res.Message.ToolCalls) == 0 {
			return nil
		}

		event := EventChat
		if *thinkingTimeStart != nil && res.Message.Content == "" && len(res.Message.ToolCalls) == 0 {
			event = EventThinking
		}
		if len(res.Message.ToolCalls) > 0 {
			event = EventToolCall
		}
		if event == EventToolCall && *thinkingTimeStart != nil && *thinkingTimeEnd == nil {
			now := time.Now()
			*thinkingTimeEnd = &now
		}
		if event == EventChat && *thinkingTimeStart != nil && *thinkingTimeEnd == nil && res.Message.Content != "" {
			now := time.Now()
			*thinkingTimeEnd = &now
		}

		json.NewEncoder(w).Encode(chatEventFromApiChatResponse(res, *thinkingTimeStart, *thinkingTimeEnd))
		flusher.Flush()

		switch event {
		case EventToolCall:
			executed, err := s.handleToolCallEvent(ctx, chat, req, res, registry, browser, thinkingTimeStart, thinkingTimeEnd, pendingAssistantToolCalls, w, flusher)
			if err != nil {
				return err
			}
			if executed {
				toolsExecuted = true
			}
		case EventChat:
			if err := s.handleChatEvent(chat, req, res, thinkingTimeStart, thinkingTimeEnd, pendingAssistantToolCalls); err != nil {
				return err
			}
		case EventThinking:
			if err := s.handleThinkingEvent(chat, req, res, thinkingTimeStart, thinkingTimeEnd, pendingAssistantToolCalls); err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
		s.log().Error("chat stream error", "error", err)
		errorEvent := s.getError(err)
		json.NewEncoder(w).Encode(errorEvent)
		flusher.Flush()
		return false, nil
	}

	return toolsExecuted, nil
}

// handleChatEvent verarbeitet Chat-Events
func (s *Server) handleChatEvent(
	chat *store.Chat,
	req responses.ChatRequest,
	res api.ChatResponse,
	thinkingTimeStart, thinkingTimeEnd **time.Time,
	pendingAssistantToolCalls *[]store.ToolCall,
) error {
	if len(chat.Messages) == 0 || chat.Messages[len(chat.Messages)-1].Role != "assistant" {
		newMsg := store.NewMessage("assistant", "", &store.MessageOptions{Model: req.Model})
		chat.Messages = append(chat.Messages, newMsg)
		if err := s.Store.AppendMessage(chat.ID, newMsg); err != nil {
			return err
		}
		if len(*pendingAssistantToolCalls) > 0 {
			lastMsg := &chat.Messages[len(chat.Messages)-1]
			lastMsg.ToolCalls = *pendingAssistantToolCalls
			*pendingAssistantToolCalls = nil
			if err := s.Store.UpdateLastMessage(chat.ID, *lastMsg); err != nil {
				return err
			}
		}
	}

	lastMsg := &chat.Messages[len(chat.Messages)-1]
	lastMsg.Content += res.Message.Content
	lastMsg.UpdatedAt = time.Now()
	if *thinkingTimeStart != nil {
		lastMsg.ThinkingTimeStart = *thinkingTimeStart
	}
	if *thinkingTimeEnd != nil {
		lastMsg.ThinkingTimeEnd = *thinkingTimeEnd
	}
	return s.Store.UpdateLastMessage(chat.ID, *lastMsg)
}

// handleThinkingEvent verarbeitet Thinking-Events
func (s *Server) handleThinkingEvent(
	chat *store.Chat,
	req responses.ChatRequest,
	res api.ChatResponse,
	thinkingTimeStart, thinkingTimeEnd **time.Time,
	pendingAssistantToolCalls *[]store.ToolCall,
) error {
	if len(chat.Messages) == 0 || chat.Messages[len(chat.Messages)-1].Role != "assistant" {
		newMsg := store.NewMessage("assistant", "", &store.MessageOptions{
			Model:    req.Model,
			Thinking: res.Message.Thinking,
		})
		chat.Messages = append(chat.Messages, newMsg)
		if err := s.Store.AppendMessage(chat.ID, newMsg); err != nil {
			return err
		}
		if len(*pendingAssistantToolCalls) > 0 {
			lastMsg := &chat.Messages[len(chat.Messages)-1]
			lastMsg.ToolCalls = *pendingAssistantToolCalls
			*pendingAssistantToolCalls = nil
			if err := s.Store.UpdateLastMessage(chat.ID, *lastMsg); err != nil {
				return err
			}
		}
	} else {
		lastMsg := &chat.Messages[len(chat.Messages)-1]
		lastMsg.Thinking += res.Message.Thinking
		lastMsg.UpdatedAt = time.Now()
		if *thinkingTimeStart != nil {
			lastMsg.ThinkingTimeStart = *thinkingTimeStart
		}
		if *thinkingTimeEnd != nil {
			lastMsg.ThinkingTimeEnd = *thinkingTimeEnd
		}
		if err := s.Store.UpdateLastMessage(chat.ID, *lastMsg); err != nil {
			return err
		}
	}
	return nil
}
