//go:build windows || darwin

// database_message.go - Message CRUD Operationen
// Enthält: getMessages, insertMessage, updateLastMessage, appendMessage

package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
)

// getMessages gibt alle Messages eines Chats zurück
func (db *database) getMessages(chatID string, loadAttachmentData bool) ([]Message, error) {
	query := `
		SELECT id, role, content, thinking, stream, model_name, created_at, updated_at, thinking_time_start, thinking_time_end, tool_result
		FROM messages
		WHERE chat_id = ?
		ORDER BY id ASC
	`

	rows, err := db.conn.Query(query, chatID)
	if err != nil {
		return nil, fmt.Errorf("query messages: %w", err)
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var msg Message
		var messageID int64
		var thinkingTimeStart, thinkingTimeEnd sql.NullTime
		var modelName sql.NullString
		var toolResult sql.NullString

		err := rows.Scan(
			&messageID,
			&msg.Role,
			&msg.Content,
			&msg.Thinking,
			&msg.Stream,
			&modelName,
			&msg.CreatedAt,
			&msg.UpdatedAt,
			&thinkingTimeStart,
			&thinkingTimeEnd,
			&toolResult,
		)
		if err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}

		attachments, err := db.getAttachments(messageID, loadAttachmentData)
		if err != nil {
			return nil, fmt.Errorf("get attachments: %w", err)
		}
		msg.Attachments = attachments

		if thinkingTimeStart.Valid {
			msg.ThinkingTimeStart = &thinkingTimeStart.Time
		}
		if thinkingTimeEnd.Valid {
			msg.ThinkingTimeEnd = &thinkingTimeEnd.Time
		}

		// Tool Result aus JSON parsen falls vorhanden
		if toolResult.Valid && toolResult.String != "" {
			var result json.RawMessage
			if err := json.Unmarshal([]byte(toolResult.String), &result); err == nil {
				msg.ToolResult = &result
			}
		}

		// Model setzen falls vorhanden
		if modelName.Valid && modelName.String != "" {
			msg.Model = modelName.String
		}

		// Tool Calls für diese Message holen
		toolCalls, err := db.getToolCalls(messageID)
		if err != nil {
			return nil, fmt.Errorf("get tool calls: %w", err)
		}
		msg.ToolCalls = toolCalls

		messages = append(messages, msg)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate messages: %w", err)
	}

	return messages, nil
}

// insertMessage fügt eine neue Message in die Datenbank ein
func (db *database) insertMessage(tx *sql.Tx, chatID string, msg Message) (int64, error) {
	query := `
		INSERT INTO messages (chat_id, role, content, thinking, stream, model_name, created_at, updated_at, thinking_time_start, thinking_time_end, tool_result)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	var thinkingTimeStart, thinkingTimeEnd sql.NullTime
	if msg.ThinkingTimeStart != nil {
		thinkingTimeStart = sql.NullTime{Time: *msg.ThinkingTimeStart, Valid: true}
	}
	if msg.ThinkingTimeEnd != nil {
		thinkingTimeEnd = sql.NullTime{Time: *msg.ThinkingTimeEnd, Valid: true}
	}

	var modelName sql.NullString
	if msg.Model != "" {
		modelName = sql.NullString{String: msg.Model, Valid: true}
	}

	var toolResultJSON sql.NullString
	if msg.ToolResult != nil {
		resultBytes, err := json.Marshal(msg.ToolResult)
		if err != nil {
			return 0, fmt.Errorf("marshal tool result: %w", err)
		}
		toolResultJSON = sql.NullString{String: string(resultBytes), Valid: true}
	}

	result, err := tx.Exec(query,
		chatID,
		msg.Role,
		msg.Content,
		msg.Thinking,
		msg.Stream,
		modelName,
		msg.CreatedAt,
		msg.UpdatedAt,
		thinkingTimeStart,
		thinkingTimeEnd,
		toolResultJSON,
	)
	if err != nil {
		return 0, err
	}

	messageID, err := result.LastInsertId()
	if err != nil {
		return 0, err
	}

	for _, att := range msg.Attachments {
		err := db.insertAttachment(tx, messageID, att)
		if err != nil {
			return 0, fmt.Errorf("insert attachment: %w", err)
		}
	}

	return messageID, nil
}

// updateLastMessage aktualisiert die letzte Message eines Chats
func (db *database) updateLastMessage(chatID string, msg Message) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// ID der letzten Message holen
	var messageID int64
	err = tx.QueryRow(`
		SELECT MAX(id) FROM messages WHERE chat_id = ?
	`, chatID).Scan(&messageID)
	if err != nil {
		return fmt.Errorf("get last message id: %w", err)
	}

	query := `
		UPDATE messages
		SET content = ?, thinking = ?, model_name = ?, updated_at = ?, thinking_time_start = ?, thinking_time_end = ?, tool_result = ?
		WHERE id = ?
	`

	var thinkingTimeStart, thinkingTimeEnd sql.NullTime
	if msg.ThinkingTimeStart != nil {
		thinkingTimeStart = sql.NullTime{Time: *msg.ThinkingTimeStart, Valid: true}
	}
	if msg.ThinkingTimeEnd != nil {
		thinkingTimeEnd = sql.NullTime{Time: *msg.ThinkingTimeEnd, Valid: true}
	}

	var modelName sql.NullString
	if msg.Model != "" {
		modelName = sql.NullString{String: msg.Model, Valid: true}
	}

	var toolResultJSON sql.NullString
	if msg.ToolResult != nil {
		resultBytes, err := json.Marshal(msg.ToolResult)
		if err != nil {
			return fmt.Errorf("marshal tool result: %w", err)
		}
		toolResultJSON = sql.NullString{String: string(resultBytes), Valid: true}
	}

	result, err := tx.Exec(query,
		msg.Content,
		msg.Thinking,
		modelName,
		msg.UpdatedAt,
		thinkingTimeStart,
		thinkingTimeEnd,
		toolResultJSON,
		messageID,
	)
	if err != nil {
		return fmt.Errorf("update last message: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("no message found to update")
	}

	_, err = tx.Exec("DELETE FROM attachments WHERE message_id = ?", messageID)
	if err != nil {
		return fmt.Errorf("delete existing attachments: %w", err)
	}
	for _, att := range msg.Attachments {
		err := db.insertAttachment(tx, messageID, att)
		if err != nil {
			return fmt.Errorf("insert attachment: %w", err)
		}
	}

	_, err = tx.Exec("DELETE FROM tool_calls WHERE message_id = ?", messageID)
	if err != nil {
		return fmt.Errorf("delete existing tool calls: %w", err)
	}
	for _, toolCall := range msg.ToolCalls {
		err := db.insertToolCall(tx, messageID, toolCall)
		if err != nil {
			return fmt.Errorf("insert tool call: %w", err)
		}
	}

	return tx.Commit()
}

// appendMessage fügt eine neue Message am Ende eines Chats hinzu
func (db *database) appendMessage(chatID string, msg Message) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	messageID, err := db.insertMessage(tx, chatID, msg)
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

	return tx.Commit()
}
