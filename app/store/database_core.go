//go:build windows || darwin

// database_core.go - Kern-Datenbank-Funktionen
// Enthält: database struct, newDatabase, Close, init, Hilfsfunktionen

package store

import (
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/mattn/go-sqlite3" // SQLite-Treiber registrieren
)

// currentSchemaVersion definiert die aktuelle Datenbank-Schema-Version.
// Wird bei Schema-Änderungen erhöht, die Migrationen erfordern.
const currentSchemaVersion = 12

// database umhüllt die SQLite-Verbindung.
// SQLite verwaltet sein eigenes Locking für konkurrierende Zugriffe:
// - Mehrere Leser können gleichzeitig auf die Datenbank zugreifen
// - Schreiber werden serialisiert (nur ein Schreiber gleichzeitig)
// - WAL-Modus erlaubt Lesern, Schreiber nicht zu blockieren
// Daher benötigen wir keine Application-Level-Locks für Datenbankoperationen.
type database struct {
	conn *sql.DB
}

// newDatabase erstellt eine neue Datenbankverbindung
func newDatabase(dbPath string) (*database, error) {
	// Datenbankverbindung öffnen
	conn, err := sql.Open("sqlite3", dbPath+"?_foreign_keys=on&_journal_mode=WAL&_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	// Verbindung testen
	if err := conn.Ping(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("ping database: %w", err)
	}

	db := &database{conn: conn}

	// Schema initialisieren
	if err := db.init(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("initialize database: %w", err)
	}

	return db, nil
}

// Close schließt die Datenbankverbindung
func (db *database) Close() error {
	_, _ = db.conn.Exec("PRAGMA wal_checkpoint(TRUNCATE);")
	return db.conn.Close()
}

// init initialisiert das Datenbankschema
func (db *database) init() error {
	if _, err := db.conn.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return fmt.Errorf("enable foreign keys: %w", err)
	}

	schema := fmt.Sprintf(`
	CREATE TABLE IF NOT EXISTS settings (
		id INTEGER PRIMARY KEY CHECK (id = 1),
		device_id TEXT NOT NULL DEFAULT '',
		has_completed_first_run BOOLEAN NOT NULL DEFAULT 0,
		expose BOOLEAN NOT NULL DEFAULT 0,
		survey BOOLEAN NOT NULL DEFAULT TRUE,
		browser BOOLEAN NOT NULL DEFAULT 0,
		models TEXT NOT NULL DEFAULT '',
		agent BOOLEAN NOT NULL DEFAULT 0,
		tools BOOLEAN NOT NULL DEFAULT 0,
		working_dir TEXT NOT NULL DEFAULT '',
		context_length INTEGER NOT NULL DEFAULT 4096,
		window_width INTEGER NOT NULL DEFAULT 0,
		window_height INTEGER NOT NULL DEFAULT 0,
		config_migrated BOOLEAN NOT NULL DEFAULT 0,
		airplane_mode BOOLEAN NOT NULL DEFAULT 0,
		turbo_enabled BOOLEAN NOT NULL DEFAULT 0,
		websearch_enabled BOOLEAN NOT NULL DEFAULT 0,
		selected_model TEXT NOT NULL DEFAULT '',
		sidebar_open BOOLEAN NOT NULL DEFAULT 0,
		think_enabled BOOLEAN NOT NULL DEFAULT 0,
		think_level TEXT NOT NULL DEFAULT '',
		remote TEXT NOT NULL DEFAULT '', -- deprecated
		schema_version INTEGER NOT NULL DEFAULT %d
	);

	-- Standard-Settings-Zeile einfügen falls nicht vorhanden
	INSERT OR IGNORE INTO settings (id) VALUES (1);

	CREATE TABLE IF NOT EXISTS chats (
		id TEXT PRIMARY KEY,
		title TEXT NOT NULL DEFAULT '',
		created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		browser_state TEXT
	);

	CREATE TABLE IF NOT EXISTS messages (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		chat_id TEXT NOT NULL,
		role TEXT NOT NULL,
		content TEXT NOT NULL DEFAULT '',
		thinking TEXT NOT NULL DEFAULT '',
		stream BOOLEAN NOT NULL DEFAULT 0,
		model_name TEXT,
		model_cloud BOOLEAN, -- deprecated
		model_ollama_host BOOLEAN, -- deprecated
		created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		thinking_time_start TIMESTAMP,
		thinking_time_end TIMESTAMP,
		tool_result TEXT,
		FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

	CREATE TABLE IF NOT EXISTS tool_calls (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		message_id INTEGER NOT NULL,
		type TEXT NOT NULL,
		function_name TEXT NOT NULL,
		function_arguments TEXT NOT NULL,
		function_result TEXT,
		FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);

	CREATE TABLE IF NOT EXISTS attachments (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		message_id INTEGER NOT NULL,
		filename TEXT NOT NULL,
		data BLOB NOT NULL,
		FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);

	CREATE TABLE IF NOT EXISTS users (
		name TEXT NOT NULL DEFAULT '',
		email TEXT NOT NULL DEFAULT '',
		plan TEXT NOT NULL DEFAULT '',
		cached_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
	`, currentSchemaVersion)

	_, err := db.conn.Exec(schema)
	if err != nil {
		return err
	}

	// Schema-Version prüfen und bei Bedarf migrieren
	if err := db.migrate(); err != nil {
		return fmt.Errorf("migrate schema: %w", err)
	}

	// Verwaiste Datensätze aufräumen (vor Foreign-Key-Constraints)
	// TODO: Kann irgendwann entfernt werden - räumt Daten vom Foreign-Key-Bug auf
	if err := db.cleanupOrphanedData(); err != nil {
		return fmt.Errorf("cleanup orphaned data: %w", err)
	}

	return nil
}

// cleanupOrphanedData entfernt verwaiste Datensätze
func (db *database) cleanupOrphanedData() error {
	_, err := db.conn.Exec(`
		DELETE FROM tool_calls
		WHERE message_id NOT IN (SELECT id FROM messages)
	`)
	if err != nil {
		return fmt.Errorf("cleanup orphaned tool_calls: %w", err)
	}

	_, err = db.conn.Exec(`
		DELETE FROM attachments
		WHERE message_id NOT IN (SELECT id FROM messages)
	`)
	if err != nil {
		return fmt.Errorf("cleanup orphaned attachments: %w", err)
	}

	_, err = db.conn.Exec(`
		DELETE FROM messages
		WHERE chat_id NOT IN (SELECT id FROM chats)
	`)
	if err != nil {
		return fmt.Errorf("cleanup orphaned messages: %w", err)
	}

	return nil
}

// duplicateColumnError prüft ob ein SQLite-Fehler eine doppelte Spalte meldet
func duplicateColumnError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "duplicate column name")
}

// columnNotExists prüft ob ein SQLite-Fehler eine fehlende Spalte meldet
func columnNotExists(err error) bool {
	return err != nil && strings.Contains(err.Error(), "no such column")
}
