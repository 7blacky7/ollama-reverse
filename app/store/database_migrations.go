//go:build windows || darwin

// database_migrations.go - Datenbank-Schema-Migrationen
// Enthält: migrate(), alle migrateVxToVy() Funktionen, Schema-Version-Handling

package store

import "fmt"

// migrate führt Datenbank-Schema-Migrationen durch
func (db *database) migrate() error {
	// Aktuelle Schema-Version holen
	version, err := db.getSchemaVersion()
	if err != nil {
		return fmt.Errorf("get schema version after migration attempt: %w", err)
	}

	// Migrationen für jede Version durchführen
	for version < currentSchemaVersion {
		switch version {
		case 1:
			// Migration von Version 1 zu 2: context_length Spalte hinzufügen
			if err := db.migrateV1ToV2(); err != nil {
				return fmt.Errorf("migrate v1 to v2: %w", err)
			}
			version = 2
		case 2:
			// Migration von Version 2 zu 3: attachments Tabelle erstellen
			if err := db.migrateV2ToV3(); err != nil {
				return fmt.Errorf("migrate v2 to v3: %w", err)
			}
			version = 3
		case 3:
			// Migration von Version 3 zu 4: tool_result Spalte zur messages Tabelle
			if err := db.migrateV3ToV4(); err != nil {
				return fmt.Errorf("migrate v3 to v4: %w", err)
			}
			version = 4
		case 4:
			// airplane_mode Spalte zur settings Tabelle hinzufügen
			if err := db.migrateV4ToV5(); err != nil {
				return fmt.Errorf("migrate v4 to v5: %w", err)
			}
			version = 5
		case 5:
			// turbo_enabled Spalte zur settings Tabelle hinzufügen
			if err := db.migrateV5ToV6(); err != nil {
				return fmt.Errorf("migrate v5 to v6: %w", err)
			}
			version = 6
		case 6:
			// Fehlenden Index für attachments Tabelle hinzufügen
			if err := db.migrateV6ToV7(); err != nil {
				return fmt.Errorf("migrate v6 to v7: %w", err)
			}
			version = 7
		case 7:
			// think_enabled und think_level Spalten zur settings Tabelle hinzufügen
			if err := db.migrateV7ToV8(); err != nil {
				return fmt.Errorf("migrate v7 to v8: %w", err)
			}
			version = 8
		case 8:
			// browser_state Spalte zur chats Tabelle hinzufügen
			if err := db.migrateV8ToV9(); err != nil {
				return fmt.Errorf("migrate v8 to v9: %w", err)
			}
			version = 9
		case 9:
			// users Tabelle hinzufügen
			if err := db.migrateV9ToV10(); err != nil {
				return fmt.Errorf("migrate v9 to v10: %w", err)
			}
			version = 10
		case 10:
			// remote Spalte aus settings Tabelle entfernen
			if err := db.migrateV10ToV11(); err != nil {
				return fmt.Errorf("migrate v10 to v11: %w", err)
			}
			version = 11
		case 11:
			// remote Spalte für Rückwärtskompatibilität zurückbringen (deprecated)
			if err := db.migrateV11ToV12(); err != nil {
				return fmt.Errorf("migrate v11 to v12: %w", err)
			}
			version = 12
		default:
			// Unbekannte Version - auf aktuell setzen
			version = currentSchemaVersion
		}
	}

	return nil
}

// migrateV1ToV2 fügt die context_length Spalte zur settings Tabelle hinzu
func (db *database) migrateV1ToV2() error {
	_, err := db.conn.Exec(`ALTER TABLE settings ADD COLUMN context_length INTEGER NOT NULL DEFAULT 4096;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add context_length column: %w", err)
	}

	_, err = db.conn.Exec(`ALTER TABLE settings ADD COLUMN survey BOOLEAN NOT NULL DEFAULT TRUE;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add survey column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 2;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}
	return nil
}

// migrateV2ToV3 erstellt die attachments Tabelle
func (db *database) migrateV2ToV3() error {
	_, err := db.conn.Exec(`
		CREATE TABLE IF NOT EXISTS attachments (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			message_id INTEGER NOT NULL,
			filename TEXT NOT NULL,
			data BLOB NOT NULL,
			FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
		)
	`)
	if err != nil {
		return fmt.Errorf("create attachments table: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 3`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV3ToV4 fügt die tool_result Spalte zur messages Tabelle hinzu
func (db *database) migrateV3ToV4() error {
	_, err := db.conn.Exec(`ALTER TABLE messages ADD COLUMN tool_result TEXT;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add tool_result column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 4;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV4ToV5 fügt die airplane_mode Spalte zur settings Tabelle hinzu
func (db *database) migrateV4ToV5() error {
	_, err := db.conn.Exec(`ALTER TABLE settings ADD COLUMN airplane_mode BOOLEAN NOT NULL DEFAULT 0;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add airplane_mode column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 5;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV5ToV6 fügt turbo_enabled, websearch_enabled, selected_model, sidebar_open Spalten hinzu
func (db *database) migrateV5ToV6() error {
	_, err := db.conn.Exec(`ALTER TABLE settings ADD COLUMN turbo_enabled BOOLEAN NOT NULL DEFAULT 0;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add turbo_enabled column: %w", err)
	}

	_, err = db.conn.Exec(`ALTER TABLE settings ADD COLUMN websearch_enabled BOOLEAN NOT NULL DEFAULT 0;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add websearch_enabled column: %w", err)
	}

	_, err = db.conn.Exec(`ALTER TABLE settings ADD COLUMN selected_model TEXT NOT NULL DEFAULT '';`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add selected_model column: %w", err)
	}

	_, err = db.conn.Exec(`ALTER TABLE settings ADD COLUMN sidebar_open BOOLEAN NOT NULL DEFAULT 0;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add sidebar_open column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 6;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV6ToV7 fügt den fehlenden Index für die attachments Tabelle hinzu
func (db *database) migrateV6ToV7() error {
	_, err := db.conn.Exec(`CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);`)
	if err != nil {
		return fmt.Errorf("create attachments index: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 7;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV7ToV8 fügt think_enabled und think_level Spalten zur settings Tabelle hinzu
func (db *database) migrateV7ToV8() error {
	_, err := db.conn.Exec(`ALTER TABLE settings ADD COLUMN think_enabled BOOLEAN NOT NULL DEFAULT 0;`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add think_enabled column: %w", err)
	}

	_, err = db.conn.Exec(`ALTER TABLE settings ADD COLUMN think_level TEXT NOT NULL DEFAULT '';`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add think_level column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 8;`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV8ToV9 fügt browser_state zur chats Tabelle hinzu
func (db *database) migrateV8ToV9() error {
	_, err := db.conn.Exec(`
		ALTER TABLE chats ADD COLUMN browser_state TEXT;
		UPDATE settings SET schema_version = 9;
	`)

	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add browser_state column: %w", err)
	}

	return nil
}

// migrateV9ToV10 erstellt die users Tabelle
func (db *database) migrateV9ToV10() error {
	_, err := db.conn.Exec(`
		CREATE TABLE IF NOT EXISTS users (
			name TEXT NOT NULL DEFAULT '',
			email TEXT NOT NULL DEFAULT '',
			plan TEXT NOT NULL DEFAULT '',
			cached_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		UPDATE settings SET schema_version = 10;
	`)
	if err != nil {
		return fmt.Errorf("create users table: %w", err)
	}

	return nil
}

// migrateV10ToV11 entfernt die remote Spalte aus der settings Tabelle
func (db *database) migrateV10ToV11() error {
	_, err := db.conn.Exec(`ALTER TABLE settings DROP COLUMN remote`)
	if err != nil && !columnNotExists(err) {
		return fmt.Errorf("drop remote column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 11`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}

// migrateV11ToV12 bringt die remote Spalte für Rückwärtskompatibilität zurück (deprecated)
func (db *database) migrateV11ToV12() error {
	_, err := db.conn.Exec(`ALTER TABLE settings ADD COLUMN remote TEXT NOT NULL DEFAULT ''`)
	if err != nil && !duplicateColumnError(err) {
		return fmt.Errorf("add remote column: %w", err)
	}

	_, err = db.conn.Exec(`UPDATE settings SET schema_version = 12`)
	if err != nil {
		return fmt.Errorf("update schema version: %w", err)
	}

	return nil
}
