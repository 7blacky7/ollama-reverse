//go:build windows || darwin

// database_settings.go - Settings und User CRUD Operationen
// Enthält: getSettings, setSettings, ID, WindowSize, ConfigMigration,
//          SchemaVersion, User-Operationen

package store

import (
	"database/sql"
	"fmt"
)

// getID gibt die Device-ID zurück
func (db *database) getID() (string, error) {
	var id string
	err := db.conn.QueryRow("SELECT device_id FROM settings").Scan(&id)
	if err != nil {
		return "", fmt.Errorf("get device id: %w", err)
	}
	return id, nil
}

// setID setzt die Device-ID
func (db *database) setID(id string) error {
	_, err := db.conn.Exec("UPDATE settings SET device_id = ?", id)
	if err != nil {
		return fmt.Errorf("set device id: %w", err)
	}
	return nil
}

// getHasCompletedFirstRun prüft ob der erste Start abgeschlossen wurde
func (db *database) getHasCompletedFirstRun() (bool, error) {
	var hasCompletedFirstRun bool
	err := db.conn.QueryRow("SELECT has_completed_first_run FROM settings").Scan(&hasCompletedFirstRun)
	if err != nil {
		return false, fmt.Errorf("get has completed first run: %w", err)
	}
	return hasCompletedFirstRun, nil
}

// setHasCompletedFirstRun setzt den First-Run-Status
func (db *database) setHasCompletedFirstRun(hasCompletedFirstRun bool) error {
	_, err := db.conn.Exec("UPDATE settings SET has_completed_first_run = ?", hasCompletedFirstRun)
	if err != nil {
		return fmt.Errorf("set has completed first run: %w", err)
	}
	return nil
}

// getSettings gibt alle Settings zurück
func (db *database) getSettings() (Settings, error) {
	var s Settings

	err := db.conn.QueryRow(`
		SELECT expose, survey, browser, models, agent, tools, working_dir, context_length, airplane_mode, turbo_enabled, websearch_enabled, selected_model, sidebar_open, think_enabled, think_level
		FROM settings
	`).Scan(&s.Expose, &s.Survey, &s.Browser, &s.Models, &s.Agent, &s.Tools, &s.WorkingDir, &s.ContextLength, &s.AirplaneMode, &s.TurboEnabled, &s.WebSearchEnabled, &s.SelectedModel, &s.SidebarOpen, &s.ThinkEnabled, &s.ThinkLevel)
	if err != nil {
		return Settings{}, fmt.Errorf("get settings: %w", err)
	}

	return s, nil
}

// setSettings speichert alle Settings
func (db *database) setSettings(s Settings) error {
	_, err := db.conn.Exec(`
		UPDATE settings
		SET expose = ?, survey = ?, browser = ?, models = ?, agent = ?, tools = ?, working_dir = ?, context_length = ?, airplane_mode = ?, turbo_enabled = ?, websearch_enabled = ?, selected_model = ?, sidebar_open = ?, think_enabled = ?, think_level = ?
	`, s.Expose, s.Survey, s.Browser, s.Models, s.Agent, s.Tools, s.WorkingDir, s.ContextLength, s.AirplaneMode, s.TurboEnabled, s.WebSearchEnabled, s.SelectedModel, s.SidebarOpen, s.ThinkEnabled, s.ThinkLevel)
	if err != nil {
		return fmt.Errorf("set settings: %w", err)
	}
	return nil
}

// getWindowSize gibt die gespeicherte Fenstergröße zurück
func (db *database) getWindowSize() (int, int, error) {
	var width, height int
	err := db.conn.QueryRow("SELECT window_width, window_height FROM settings").Scan(&width, &height)
	if err != nil {
		return 0, 0, fmt.Errorf("get window size: %w", err)
	}
	return width, height, nil
}

// setWindowSize speichert die Fenstergröße
func (db *database) setWindowSize(width, height int) error {
	_, err := db.conn.Exec("UPDATE settings SET window_width = ?, window_height = ?", width, height)
	if err != nil {
		return fmt.Errorf("set window size: %w", err)
	}
	return nil
}

// isConfigMigrated prüft ob die Config-Migration durchgeführt wurde
func (db *database) isConfigMigrated() (bool, error) {
	var migrated bool
	err := db.conn.QueryRow("SELECT config_migrated FROM settings").Scan(&migrated)
	if err != nil {
		return false, fmt.Errorf("get config migrated: %w", err)
	}
	return migrated, nil
}

// setConfigMigrated setzt den Config-Migration-Status
func (db *database) setConfigMigrated(migrated bool) error {
	_, err := db.conn.Exec("UPDATE settings SET config_migrated = ?", migrated)
	if err != nil {
		return fmt.Errorf("set config migrated: %w", err)
	}
	return nil
}

// getSchemaVersion gibt die aktuelle Schema-Version zurück
func (db *database) getSchemaVersion() (int, error) {
	var version int
	err := db.conn.QueryRow("SELECT schema_version FROM settings").Scan(&version)
	if err != nil {
		return 0, fmt.Errorf("get schema version: %w", err)
	}
	return version, nil
}

// setSchemaVersion setzt die Schema-Version
func (db *database) setSchemaVersion(version int) error {
	_, err := db.conn.Exec("UPDATE settings SET schema_version = ?", version)
	if err != nil {
		return fmt.Errorf("set schema version: %w", err)
	}
	return nil
}

// getUser gibt den gecachten User zurück
func (db *database) getUser() (*User, error) {
	var user User
	err := db.conn.QueryRow(`
		SELECT name, email, plan, cached_at
		FROM users
		LIMIT 1
	`).Scan(&user.Name, &user.Email, &user.Plan, &user.CachedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Kein User gecacht
		}
		return nil, fmt.Errorf("get user: %w", err)
	}

	return &user, nil
}

// setUser speichert einen User im Cache
func (db *database) setUser(user User) error {
	if err := db.clearUser(); err != nil {
		return fmt.Errorf("before set: %w", err)
	}

	_, err := db.conn.Exec(`
		INSERT INTO users (name, email, plan, cached_at)
		VALUES (?, ?, ?, ?)
	`, user.Name, user.Email, user.Plan, user.CachedAt)
	if err != nil {
		return fmt.Errorf("set user: %w", err)
	}

	return nil
}

// clearUser löscht den gecachten User
func (db *database) clearUser() error {
	_, err := db.conn.Exec("DELETE FROM users")
	if err != nil {
		return fmt.Errorf("clear user: %w", err)
	}
	return nil
}
