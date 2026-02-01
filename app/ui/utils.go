//go:build windows || darwin

// utils.go - Hilfsfunktionen und Utilities
// Enthält: getError, isNetworkError, userAgent, kleine Hilfsfunktionen

package ui

import (
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/ui/responses"
)

// getError konvertiert Fehler zu ErrorEvent
func (s *Server) getError(err error) responses.ErrorEvent {
	var sErr api.AuthorizationError
	if errors.As(err, &sErr) && sErr.StatusCode == http.StatusUnauthorized {
		return responses.ErrorEvent{
			EventName: "error",
			Error:     "Could not verify you are signed in. Please sign in and try again.",
			Code:      "cloud_unauthorized",
		}
	}

	errStr := err.Error()

	switch {
	case strings.Contains(errStr, "402"):
		return responses.ErrorEvent{
			EventName: "error",
			Error:     "You've reached your usage limit, please upgrade to continue",
			Code:      "usage_limit_upgrade",
		}
	case strings.HasPrefix(errStr, "pull model manifest") && isNetworkError(errStr):
		return responses.ErrorEvent{
			EventName: "error",
			Error:     "Unable to download model. Please check your internet connection to download the model for offline use.",
			Code:      "offline_download_error",
		}
	case errors.Is(err, ErrNetworkOffline) || strings.Contains(errStr, "operation timed out"):
		return responses.ErrorEvent{
			EventName: "error",
			Error:     "Connection lost",
			Code:      "turbo_connection_lost",
		}
	}
	return responses.ErrorEvent{
		EventName: "error",
		Error:     err.Error(),
	}
}

// isNetworkError prüft ob ein Fehler ein Netzwerk-Fehler ist
func isNetworkError(errStr string) bool {
	networkErrorPatterns := []string{
		"connection refused",
		"no such host",
		"timeout",
		"network is unreachable",
		"connection reset",
		"connection timed out",
		"temporary failure",
		"dial tcp",
		"i/o timeout",
		"context deadline exceeded",
		"broken pipe",
	}

	for _, pattern := range networkErrorPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}
	return false
}

// userAgent erstellt den User-Agent-String
func userAgent() string {
	buildinfo, _ := debug.ReadBuildInfo()

	version := buildinfo.Main.Version
	if version == "(devel)" {
		version = "v0.0.0"
	}

	return fmt.Sprintf("ollama/%s (%s %s) app/%s Go/%s",
		version,
		runtime.GOARCH,
		runtime.GOOS,
		version,
		runtime.Version(),
	)
}

// getStringFromMap holt einen String sicher aus einer Map
func getStringFromMap(m map[string]any, key, defaultValue string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return defaultValue
}

// isImageAttachment prüft ob ein Dateiname ein Bild ist
func isImageAttachment(filename string) bool {
	ext := strings.ToLower(filename)
	return strings.HasSuffix(ext, ".png") || strings.HasSuffix(ext, ".jpg") || strings.HasSuffix(ext, ".jpeg") || strings.HasSuffix(ext, ".webp")
}

// ptr ist eine Hilfsfunktion für Pointer auf Literals
func ptr[T any](v T) *T { return &v }

// supportsBrowserTools prüft ob ein Model Browser-Tools unterstützt
func supportsBrowserTools(model string) bool {
	return strings.HasPrefix(strings.ToLower(model), "gpt-oss")
}

// supportsWebSearchTools prüft ob ein Model Web-Search-Tools unterstützt
func supportsWebSearchTools(model string) bool {
	model = strings.ToLower(model)
	prefixes := []string{"qwen3", "deepseek-v3"}
	for _, p := range prefixes {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

// splitModelName teilt einen Model-Namen in Name und Tag
func splitModelName(modelName string) []string {
	return strings.Split(modelName, ":")
}

// containsSlash prüft ob ein String einen Slash enthält
func containsSlash(s string) bool {
	return strings.Contains(s, "/")
}
