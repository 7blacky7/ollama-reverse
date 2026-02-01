//go:build windows || darwin

// app_windows_shortcut.go - Windows Shortcut-Verwaltung
//
// Enthaelt:
// - createLoginShortcut: Erstellt Autostart-Verknuepfung
package main

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

// createLoginShortcut erstellt die Autostart-Verknuepfung fuer Windows
func createLoginShortcut() error {
	// The installer lays down a shortcut for us so we can copy it without
	// having to resort to calling COM APIs to establish the shortcut
	shortcutOrigin := filepath.Join(appPath, "lib", "Ollama.lnk")

	_, err := os.Stat(startupShortcut)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			in, err := os.Open(shortcutOrigin)
			if err != nil {
				return fmt.Errorf("unable to open shortcut %s : %w", shortcutOrigin, err)
			}
			defer in.Close()
			out, err := os.Create(startupShortcut)
			if err != nil {
				return fmt.Errorf("unable to open startup link %s : %w", startupShortcut, err)
			}
			defer out.Close()
			_, err = io.Copy(out, in)
			if err != nil {
				return fmt.Errorf("unable to copy shortcut %s : %w", startupShortcut, err)
			}
			err = out.Sync()
			if err != nil {
				return fmt.Errorf("unable to sync shortcut %s : %w", startupShortcut, err)
			}
			slog.Info("Created Startup shortcut", "shortcut", startupShortcut)
		} else {
			slog.Warn("unexpected error looking up Startup shortcut", "error", err)
		}
	} else {
		slog.Debug("Startup link already exists", "shortcut", startupShortcut)
	}
	return nil
}
