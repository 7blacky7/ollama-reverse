//go:build windows || darwin

// database_chat.go - Chat CRUD Operationen
// Enthält: getAllChats, getChatWithOptions, saveChat, deleteChat, updateChatBrowserState

package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

// getAllChats gibt alle Chats mit erstem User-Message zurück
func (db *database) getAllChats() ([]Chat, error) {
	// Chats mit erstem User-Message und letzter Update-Zeit abfragen
	query := `
		SELECT
			c.id,
			c.title,
			c.created_at,
			COALESCE(first_msg.content, '') as first_user_content,
			COALESCE(datetime(MAX(m.updated_at)), datetime(c.created_at)) as last_updated
		FROM chats c
		LEFT JOIN (
			SELECT chat_id, content, MIN(id) as min_id
			FROM messages
			WHERE role = 'user'
			GROUP BY chat_id
		) first_msg ON c.id = first_msg.chat_id
		LEFT JOIN messages m ON c.id = m.chat_id
		GROUP BY c.id, c.title, c.created_at, first_msg.content
		ORDER BY last_updated DESC
	`

	rows, err := db.conn.Query(query)
	if err != nil {
		return nil, fmt.Errorf("query chats: %w", err)
	}
	defer rows.Close()

	var chats []Chat
	for rows.Next() {
		var chat Chat
		var createdAt time.Time
		var firstUserContent string
		var lastUpdatedStr string

		err := rows.Scan(
			&chat.ID,
			&chat.Title,
			&createdAt,
			&firstUserContent,
			&lastUpdatedStr,
		)

		// Letzte Update-Zeit parsen
		lastUpdated, _ := time.Parse("2006-01-02 15:04:05", lastUpdatedStr)
		if err != nil {
			return nil, fmt.Errorf("scan chat: %w", err)
		}

		chat.CreatedAt = createdAt

		// Dummy-User-Message für UI-Anzeige hinzufügen
		// Nur für Excerpt, vollständige Messages werden bei Bedarf geladen
		chat.Messages = []Message{}
		if firstUserContent != "" {
			chat.Messages = append(chat.Messages, Message{
				Role:      "user",
				Content:   firstUserContent,
				UpdatedAt: lastUpdated,
			})
		}

		chats = append(chats, chat)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate chats: %w", err)
	}

	return chats, nil
}

// getChatWithOptions gibt einen Chat mit optionalen Attachment-Daten zurück
func (db *database) getChatWithOptions(id string, loadAttachmentData bool) (*Chat, error) {
	query := `
		SELECT id, title, created_at, browser_state
		FROM chats
		WHERE id = ?
	`

	var chat Chat
	var createdAt time.Time
	var browserState sql.NullString

	err := db.conn.QueryRow(query, id).Scan(
		&chat.ID,
		&chat.Title,
		&createdAt,
		&browserState,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("chat not found")
		}
		return nil, fmt.Errorf("query chat: %w", err)
	}

	chat.CreatedAt = createdAt
	if browserState.Valid && browserState.String != "" {
		var raw json.RawMessage
		if err := json.Unmarshal([]byte(browserState.String), &raw); err == nil {
			chat.BrowserState = raw
		}
	}

	messages, err := db.getMessages(id, loadAttachmentData)
	if err != nil {
		return nil, fmt.Errorf("get messages: %w", err)
	}
	chat.Messages = messages

	return &chat, nil
}

// saveChat speichert einen Chat mit allen Messages
func (db *database) saveChat(chat Chat) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// COALESCE für browser_state um bestehenden Wert nicht zu überschreiben
	// wenn kein neuer State mitgegeben wird
	query := `
		INSERT INTO chats (id, title, created_at, browser_state)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			title = excluded.title,
			browser_state = COALESCE(excluded.browser_state, chats.browser_state)
	`

	var browserState sql.NullString
	if chat.BrowserState != nil {
		browserState = sql.NullString{String: string(chat.BrowserState), Valid: true}
	}

	_, err = tx.Exec(query,
		chat.ID,
		chat.Title,
		chat.CreatedAt,
		browserState,
	)
	if err != nil {
		return fmt.Errorf("save chat: %w", err)
	}

	// Bestehende Messages löschen (werden alle neu eingefügt)
	_, err = tx.Exec("DELETE FROM messages WHERE chat_id = ?", chat.ID)
	if err != nil {
		return fmt.Errorf("delete messages: %w", err)
	}

	// Messages einfügen
	for _, msg := range chat.Messages {
		messageID, err := db.insertMessage(tx, chat.ID, msg)
		if err != nil {
			return fmt.Errorf("insert message: %w", err)
		}

		// Tool Calls einfügen falls vorhanden
		for _, toolCall := range msg.ToolCalls {
			err := db.insertToolCall(tx, messageID, toolCall)
			if err != nil {
				return fmt.Errorf("insert tool call: %w", err)
			}
		}
	}

	return tx.Commit()
}

// updateChatBrowserState aktualisiert nur den browser_state eines Chats
func (db *database) updateChatBrowserState(chatID string, state json.RawMessage) error {
	_, err := db.conn.Exec(`UPDATE chats SET browser_state = ? WHERE id = ?`, string(state), chatID)
	if err != nil {
		return fmt.Errorf("update chat browser state: %w", err)
	}
	return nil
}

// deleteChat löscht einen Chat und alle zugehörigen Daten
func (db *database) deleteChat(id string) error {
	_, err := db.conn.Exec("DELETE FROM chats WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("delete chat: %w", err)
	}

	_, _ = db.conn.Exec("PRAGMA wal_checkpoint(TRUNCATE);")

	return nil
}
