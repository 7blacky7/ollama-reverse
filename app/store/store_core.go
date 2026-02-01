//go:build windows || darwin

// Modul: store_core.go
// Beschreibung: Store-Kernfunktionen und Datenbank-Initialisierung.
// Enthaelt ensureDB, Migration und Basiskonfiguration.

package store

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/google/uuid"
)

type Store struct {
	// DBPath allows overriding the default database path (mainly for testing)
	DBPath string

	// dbMu protects database initialization only
	dbMu sync.Mutex
	db   *database
}

var defaultDBPath = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "db.sqlite")
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Application Support", "Ollama", "db.sqlite")
	default:
		return filepath.Join(os.Getenv("HOME"), ".ollama", "db.sqlite")
	}
}()

// legacyConfigPath is the path to the old config.json file
var legacyConfigPath = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "config.json")
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Application Support", "Ollama", "config.json")
	default:
		return filepath.Join(os.Getenv("HOME"), ".ollama", "config.json")
	}
}()

// legacyData represents the old config.json structure (only fields we need to migrate)
type legacyData struct {
	ID           string `json:"id"`
	FirstTimeRun bool   `json:"first-time-run"`
}

func (s *Store) ensureDB() error {
	// Fast path: check if db is already initialized
	if s.db != nil {
		return nil
	}

	// Slow path: initialize database with lock
	s.dbMu.Lock()
	defer s.dbMu.Unlock()

	// Double-check after acquiring lock
	if s.db != nil {
		return nil
	}

	dbPath := s.DBPath
	if dbPath == "" {
		dbPath = defaultDBPath
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return fmt.Errorf("create db directory: %w", err)
	}

	database, err := newDatabase(dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}

	// Generate device ID if needed
	id, err := database.getID()
	if err != nil || id == "" {
		// Generate new UUID for device
		u, err := uuid.NewV7()
		if err == nil {
			database.setID(u.String())
		}
	}

	s.db = database

	// Check if we need to migrate from config.json
	migrated, err := database.isConfigMigrated()
	if err != nil || !migrated {
		if err := s.migrateFromConfig(database); err != nil {
			slog.Warn("failed to migrate from config.json", "error", err)
		}
	}

	return nil
}

// migrateFromConfig attempts to migrate ID and FirstTimeRun from config.json
func (s *Store) migrateFromConfig(database *database) error {
	configPath := legacyConfigPath

	// Check if config.json exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// No config to migrate, mark as migrated
		return database.setConfigMigrated(true)
	}

	// Read the config file
	b, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read legacy config: %w", err)
	}

	var legacy legacyData
	if err := json.Unmarshal(b, &legacy); err != nil {
		// If we can't parse it, just mark as migrated and move on
		slog.Warn("failed to parse legacy config.json", "error", err)
		return database.setConfigMigrated(true)
	}

	// Migrate the ID if present
	if legacy.ID != "" {
		if err := database.setID(legacy.ID); err != nil {
			return fmt.Errorf("migrate device ID: %w", err)
		}
		slog.Info("migrated device ID from config.json")
	}

	hasCompleted := legacy.FirstTimeRun // If old FirstTimeRun is true, it means first run was completed
	if err := database.setHasCompletedFirstRun(hasCompleted); err != nil {
		return fmt.Errorf("migrate first time run: %w", err)
	}
	slog.Info("migrated first run status from config.json", "hasCompleted", hasCompleted)

	// Mark as migrated
	if err := database.setConfigMigrated(true); err != nil {
		return fmt.Errorf("mark config as migrated: %w", err)
	}

	slog.Info("successfully migrated settings from config.json")
	return nil
}

func (s *Store) ID() (string, error) {
	if err := s.ensureDB(); err != nil {
		return "", err
	}

	return s.db.getID()
}

func (s *Store) HasCompletedFirstRun() (bool, error) {
	if err := s.ensureDB(); err != nil {
		return false, err
	}

	return s.db.getHasCompletedFirstRun()
}

func (s *Store) SetHasCompletedFirstRun(hasCompleted bool) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setHasCompletedFirstRun(hasCompleted)
}

func (s *Store) Close() error {
	s.dbMu.Lock()
	defer s.dbMu.Unlock()

	if s.db != nil {
		return s.db.Close()
	}
	return nil
}
