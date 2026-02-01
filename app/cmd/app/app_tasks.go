//go:build windows || darwin

// Package main - Hintergrund-Tasks und Hilfsfunktionen.
// Dieses Modul enthaelt Funktionen fuer Updates und Login-Pruefungen.

package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/ollama/ollama/app/updater"
)

// startHiddenTasks fuehrt Tasks im Hintergrund aus, wenn die App versteckt startet.
func startHiddenTasks() {
	// If an upgrade is ready and we're in hidden mode, perform it at startup.
	// If we're not in hidden mode, we want to start as fast as possible and not
	// slow the user down with an upgrade.
	if updater.IsUpdatePending() {
		if fastStartup {
			// CLI triggered app startup use-case
			slog.Info("deferring pending update for fast startup")
		} else {
			if err := updater.DoUpgradeAtStartup(); err != nil {
				slog.Info("unable to perform upgrade at startup", "error", err)
				// Make sure the restart to upgrade menu shows so we can attempt an interactive upgrade to get authorization
				UpdateAvailable("")
			} else {
				slog.Debug("launching new version...")
				// TODO - consider a timer that aborts if this takes too long and we haven't been killed yet...
				LaunchNewApp()
				os.Exit(0)
			}
		}
	}
}

// checkUserLoggedIn prueft, ob ein Benutzer eingeloggt ist.
func checkUserLoggedIn(uiServerPort int) bool {
	if uiServerPort == 0 {
		slog.Debug("UI server not ready yet, skipping auth check")
		return false
	}

	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/api/me", uiServerPort), "application/json", nil)
	if err != nil {
		slog.Debug("failed to call local auth endpoint", "error", err)
		return false
	}
	defer resp.Body.Close()

	// Check if the response is successful
	if resp.StatusCode != http.StatusOK {
		slog.Debug("auth endpoint returned non-OK status", "status", resp.StatusCode)
		return false
	}

	var user struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		slog.Debug("failed to parse user response", "error", err)
		return false
	}

	// Verify we have a valid user with an ID and name
	if user.ID == "" || user.Name == "" {
		slog.Debug("user response missing required fields", "id", user.ID, "name", user.Name)
		return false
	}

	slog.Debug("user is logged in", "user_id", user.ID, "user_name", user.Name)
	return true
}
