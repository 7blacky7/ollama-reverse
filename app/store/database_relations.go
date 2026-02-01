//go:build windows || darwin

// database_relations.go - Attachment und ToolCall Operationen
// Enthält: getAttachments, getToolCalls, insertAttachment, insertToolCall

package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
)

// getAttachments gibt alle Attachments einer Message zurück
func (db *database) getAttachments(messageID int64, loadData bool) ([]File, error) {
	var query string
	if loadData {
		query = `
			SELECT filename, data
			FROM attachments
			WHERE message_id = ?
			ORDER BY id ASC
		`
	} else {
		query = `
			SELECT filename, '' as data
			FROM attachments
			WHERE message_id = ?
			ORDER BY id ASC
		`
	}

	rows, err := db.conn.Query(query, messageID)
	if err != nil {
		return nil, fmt.Errorf("query attachments: %w", err)
	}
	defer rows.Close()

	var attachments []File
	for rows.Next() {
		var file File
		err := rows.Scan(&file.Filename, &file.Data)
		if err != nil {
			return nil, fmt.Errorf("scan attachment: %w", err)
		}
		attachments = append(attachments, file)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate attachments: %w", err)
	}

	return attachments, nil
}

// getToolCalls gibt alle Tool Calls einer Message zurück
func (db *database) getToolCalls(messageID int64) ([]ToolCall, error) {
	query := `
		SELECT type, function_name, function_arguments, function_result
		FROM tool_calls
		WHERE message_id = ?
		ORDER BY id ASC
	`

	rows, err := db.conn.Query(query, messageID)
	if err != nil {
		return nil, fmt.Errorf("query tool calls: %w", err)
	}
	defer rows.Close()

	var toolCalls []ToolCall
	for rows.Next() {
		var tc ToolCall
		var functionResult sql.NullString

		err := rows.Scan(
			&tc.Type,
			&tc.Function.Name,
			&tc.Function.Arguments,
			&functionResult,
		)
		if err != nil {
			return nil, fmt.Errorf("scan tool call: %w", err)
		}

		if functionResult.Valid && functionResult.String != "" {
			// JSON Result parsen
			var result json.RawMessage
			if err := json.Unmarshal([]byte(functionResult.String), &result); err == nil {
				tc.Function.Result = &result
			}
		}

		toolCalls = append(toolCalls, tc)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate tool calls: %w", err)
	}

	return toolCalls, nil
}

// insertAttachment fügt ein Attachment in die Datenbank ein
func (db *database) insertAttachment(tx *sql.Tx, messageID int64, file File) error {
	query := `
		INSERT INTO attachments (message_id, filename, data)
		VALUES (?, ?, ?)
	`
	_, err := tx.Exec(query, messageID, file.Filename, file.Data)
	return err
}

// insertToolCall fügt einen Tool Call in die Datenbank ein
func (db *database) insertToolCall(tx *sql.Tx, messageID int64, tc ToolCall) error {
	query := `
		INSERT INTO tool_calls (message_id, type, function_name, function_arguments, function_result)
		VALUES (?, ?, ?, ?, ?)
	`

	var functionResult sql.NullString
	if tc.Function.Result != nil {
		// Result zu JSON konvertieren
		resultJSON, err := json.Marshal(tc.Function.Result)
		if err != nil {
			return fmt.Errorf("marshal tool result: %w", err)
		}
		functionResult = sql.NullString{String: string(resultJSON), Valid: true}
	}

	_, err := tx.Exec(query,
		messageID,
		tc.Type,
		tc.Function.Name,
		tc.Function.Arguments,
		functionResult,
	)
	return err
}
