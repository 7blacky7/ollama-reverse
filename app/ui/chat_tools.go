//go:build windows || darwin

// chat_tools.go - Tool-Ausführung und Event-Handling
// Enthält: executeToolCall, handleToolError, handleToolCallEvent

package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui/responses"
)

// handleToolCallEvent verarbeitet Tool-Call-Events
func (s *Server) handleToolCallEvent(
	ctx context.Context,
	chat *store.Chat,
	req responses.ChatRequest,
	res api.ChatResponse,
	registry *tools.Registry,
	browser *tools.Browser,
	thinkingTimeStart, thinkingTimeEnd **time.Time,
	pendingAssistantToolCalls *[]store.ToolCall,
	w http.ResponseWriter,
	flusher http.Flusher,
) (bool, error) {
	toolsExecuted := false

	if *thinkingTimeEnd != nil {
		if len(chat.Messages) > 0 && chat.Messages[len(chat.Messages)-1].Role == "assistant" {
			lastMsg := &chat.Messages[len(chat.Messages)-1]
			lastMsg.ThinkingTimeEnd = *thinkingTimeEnd
			lastMsg.UpdatedAt = time.Now()
			s.Store.UpdateLastMessage(chat.ID, *lastMsg)
		}
		*thinkingTimeStart = nil
		*thinkingTimeEnd = nil
	}

	// Tool-Calls an Assistant anhängen
	if len(res.Message.ToolCalls) > 0 {
		if len(chat.Messages) > 0 && chat.Messages[len(chat.Messages)-1].Role == "assistant" {
			toolCalls := convertToolCalls(res.Message.ToolCalls)
			lastMsg := &chat.Messages[len(chat.Messages)-1]
			lastMsg.ToolCalls = toolCalls
			if err := s.Store.UpdateLastMessage(chat.ID, *lastMsg); err != nil {
				return false, err
			}
		} else {
			// Standalone web_search/web_fetch
			onlyStandalone := true
			for _, tc := range res.Message.ToolCalls {
				if !(tc.Function.Name == "web_search" || tc.Function.Name == "web_fetch") {
					onlyStandalone = false
					break
				}
			}
			if onlyStandalone {
				toolCalls := convertToolCalls(res.Message.ToolCalls)
				synth := store.NewMessage("assistant", "", &store.MessageOptions{Model: req.Model, ToolCalls: toolCalls})
				chat.Messages = append(chat.Messages, synth)
				if err := s.Store.AppendMessage(chat.ID, synth); err != nil {
					return false, err
				}
				*pendingAssistantToolCalls = nil
			}
		}
	}

	// Tools ausführen
	for _, toolCall := range res.Message.ToolCalls {
		toolsExecuted = true
		if err := s.executeToolCall(ctx, chat, toolCall, registry, browser, w, flusher); err != nil {
			return toolsExecuted, err
		}
	}

	return toolsExecuted, nil
}

// executeToolCall führt einen einzelnen Tool-Call aus
func (s *Server) executeToolCall(
	ctx context.Context,
	chat *store.Chat,
	toolCall api.ToolCall,
	registry *tools.Registry,
	browser *tools.Browser,
	w http.ResponseWriter,
	flusher http.Flusher,
) error {
	result, content, err := registry.Execute(ctx, toolCall.Function.Name, toolCall.Function.Arguments.ToMap())
	if err != nil {
		return s.handleToolError(chat, toolCall, err, w, flusher)
	}

	var tr json.RawMessage
	if strings.HasPrefix(toolCall.Function.Name, "browser.search") {
		// Standalone web_search braucht keinen Browser-State
	} else if strings.HasPrefix(toolCall.Function.Name, "browser") {
		stateBytes, err := json.Marshal(browser.State())
		if err != nil {
			return fmt.Errorf("failed to marshal browser state: %w", err)
		}
		if err := s.Store.UpdateChatBrowserState(chat.ID, json.RawMessage(stateBytes)); err != nil {
			return fmt.Errorf("failed to persist browser state to chat: %w", err)
		}
	} else {
		var err error
		tr, err = json.Marshal(result)
		if err != nil {
			return fmt.Errorf("failed to marshal tool result: %w", err)
		}
	}

	// Content für Model sicherstellen
	modelContent := content
	if toolCall.Function.Name == "web_fetch" && modelContent == "" {
		if str, ok := result.(string); ok {
			modelContent = str
		}
	}
	if modelContent == "" && len(tr) > 0 {
		s.log().Debug("tool message empty, sending json result")
		modelContent = string(tr)
	}

	toolMsg := store.NewMessage("tool", modelContent, &store.MessageOptions{
		ToolResult: &tr,
	})
	toolMsg.ToolName = toolCall.Function.Name
	chat.Messages = append(chat.Messages, toolMsg)
	s.Store.AppendMessage(chat.ID, toolMsg)

	// Events senden
	toolResult := true
	json.NewEncoder(w).Encode(responses.ChatEvent{
		EventName: "tool",
		Content:   &content,
		ToolName:  &toolCall.Function.Name,
	})
	flusher.Flush()

	var toolState any = nil
	if browser != nil {
		toolState = browser.State()
	}

	json.NewEncoder(w).Encode(responses.ChatEvent{
		EventName:      "tool_result",
		Content:        &content,
		ToolName:       &toolCall.Function.Name,
		ToolResult:     &toolResult,
		ToolResultData: result,
		ToolState:      toolState,
	})
	flusher.Flush()

	return nil
}

// handleToolError behandelt Tool-Fehler
func (s *Server) handleToolError(chat *store.Chat, toolCall api.ToolCall, err error, w http.ResponseWriter, flusher http.Flusher) error {
	errContent := fmt.Sprintf("Error: %v", err)
	toolErrMsg := store.NewMessage("tool", errContent, nil)
	toolErrMsg.ToolName = toolCall.Function.Name
	chat.Messages = append(chat.Messages, toolErrMsg)
	if err := s.Store.AppendMessage(chat.ID, toolErrMsg); err != nil {
		return err
	}

	toolResult := true
	json.NewEncoder(w).Encode(responses.ChatEvent{
		EventName: "tool",
		Content:   &errContent,
		ToolName:  &toolCall.Function.Name,
	})
	flusher.Flush()

	json.NewEncoder(w).Encode(responses.ChatEvent{
		EventName:      "tool_result",
		Content:        &errContent,
		ToolName:       &toolCall.Function.Name,
		ToolResult:     &toolResult,
		ToolResultData: nil,
	})
	flusher.Flush()

	return nil
}

// convertToolCalls konvertiert API ToolCalls zu Store ToolCalls
func convertToolCalls(apiCalls []api.ToolCall) []store.ToolCall {
	toolCalls := make([]store.ToolCall, len(apiCalls))
	for i, tc := range apiCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		toolCalls[i] = store.ToolCall{
			Type: "function",
			Function: store.ToolFunction{
				Name:      tc.Function.Name,
				Arguments: string(argsJSON),
			},
		}
	}
	return toolCalls
}
