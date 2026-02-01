//go:build windows || darwin

// Modul: store_operations.go
// Beschreibung: Store-Operationen fuer Settings, Chats und User.
// Enthaelt alle CRUD-Operationen auf der Datenbank.

package store

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/app/types/not"
)

func (s *Store) Settings() (Settings, error) {
	if err := s.ensureDB(); err != nil {
		return Settings{}, fmt.Errorf("load settings: %w", err)
	}

	settings, err := s.db.getSettings()
	if err != nil {
		return Settings{}, err
	}

	// Set default models directory if not set
	if settings.Models == "" {
		dir := os.Getenv("OLLAMA_MODELS")
		if dir != "" {
			settings.Models = dir
		} else {
			home, err := os.UserHomeDir()
			if err == nil {
				settings.Models = filepath.Join(home, ".ollama", "models")
			}
		}
	}

	return settings, nil
}

func (s *Store) SetSettings(settings Settings) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setSettings(settings)
}

func (s *Store) Chats() ([]Chat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	return s.db.getAllChats()
}

func (s *Store) Chat(id string) (*Chat, error) {
	return s.ChatWithOptions(id, true)
}

func (s *Store) ChatWithOptions(id string, loadAttachmentData bool) (*Chat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	chat, err := s.db.getChatWithOptions(id, loadAttachmentData)
	if err != nil {
		return nil, fmt.Errorf("%w: chat %s", not.Found, id)
	}

	return chat, nil
}

func (s *Store) SetChat(chat Chat) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.saveChat(chat)
}

func (s *Store) DeleteChat(id string) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	// Delete from database
	if err := s.db.deleteChat(id); err != nil {
		return fmt.Errorf("%w: chat %s", not.Found, id)
	}

	// Also delete associated images
	chatImgDir := filepath.Join(s.ImgDir(), id)
	if err := os.RemoveAll(chatImgDir); err != nil {
		// Log error but don't fail the deletion
		slog.Warn("failed to delete chat images", "chat_id", id, "error", err)
	}

	return nil
}

func (s *Store) WindowSize() (int, int, error) {
	if err := s.ensureDB(); err != nil {
		return 0, 0, err
	}

	return s.db.getWindowSize()
}

func (s *Store) SetWindowSize(width, height int) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setWindowSize(width, height)
}

func (s *Store) UpdateLastMessage(chatID string, message Message) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.updateLastMessage(chatID, message)
}

func (s *Store) AppendMessage(chatID string, message Message) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.appendMessage(chatID, message)
}

func (s *Store) UpdateChatBrowserState(chatID string, state json.RawMessage) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.updateChatBrowserState(chatID, state)
}

func (s *Store) User() (*User, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	return s.db.getUser()
}

func (s *Store) SetUser(user User) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	user.CachedAt = time.Now()
	return s.db.setUser(user)
}

func (s *Store) ClearUser() error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.clearUser()
}
