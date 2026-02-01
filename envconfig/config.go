// config.go - Haupt-Konfigurationsfunktionen fuer Ollama
//
// Dieses Modul enthaelt:
// - Host: Gibt Scheme und Host zurueck (OLLAMA_HOST)
// - AllowedOrigins: Gibt erlaubte Origins zurueck (OLLAMA_ORIGINS)
// - Models: Gibt Model-Verzeichnis zurueck (OLLAMA_MODELS)
// - KeepAlive: Gibt Keep-Alive-Dauer zurueck (OLLAMA_KEEP_ALIVE)
// - LoadTimeout: Gibt Load-Timeout zurueck (OLLAMA_LOAD_TIMEOUT)
// - Remotes: Gibt erlaubte Remote-Hosts zurueck (OLLAMA_REMOTES)
// - LogLevel: Gibt Log-Level zurueck (OLLAMA_DEBUG)
//
// Weitere Konfigurationen sind ausgelagert:
// - config_features.go: Feature-Flags und GPU-Variablen
// - config_utils.go: Utility-Funktionen und AsMap/Values
package envconfig

import (
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Host gibt Scheme und Host zurueck
// Konfigurierbar via OLLAMA_HOST
// Default: http://127.0.0.1:11434
func Host() *url.URL {
	defaultPort := "11434"

	s := strings.TrimSpace(Var("OLLAMA_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
		if s == "ollama.com" {
			scheme, hostport = "https", "ollama.com:443"
		}
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// AllowedOrigins gibt erlaubte Origins zurueck
// Konfigurierbar via OLLAMA_ORIGINS (komma-separiert)
// Enthaelt Standard-Origins fuer localhost
func AllowedOrigins() (origins []string) {
	if s := Var("OLLAMA_ORIGINS"); s != "" {
		origins = strings.Split(s, ",")
	}

	// Standard-Origins fuer localhost
	for _, origin := range []string{"localhost", "127.0.0.1", "0.0.0.0"} {
		origins = append(origins,
			fmt.Sprintf("http://%s", origin),
			fmt.Sprintf("https://%s", origin),
			fmt.Sprintf("http://%s", net.JoinHostPort(origin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(origin, "*")),
		)
	}

	// App-Protokolle
	origins = append(origins,
		"app://*",
		"file://*",
		"tauri://*",
		"vscode-webview://*",
		"vscode-file://*",
	)

	return origins
}

// Models gibt das Model-Verzeichnis zurueck
// Konfigurierbar via OLLAMA_MODELS
// Default: $HOME/.ollama/models
func Models() string {
	if s := Var("OLLAMA_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama", "models")
}

// KeepAlive gibt die Dauer zurueck, die Models im Speicher bleiben
// Konfigurierbar via OLLAMA_KEEP_ALIVE
// Negative Werte = unendlich, 0 = kein Keep-Alive
// Default: 5 Minuten
func KeepAlive() (keepAlive time.Duration) {
	keepAlive = 5 * time.Minute
	if s := Var("OLLAMA_KEEP_ALIVE"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			keepAlive = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			keepAlive = time.Duration(n) * time.Second
		}
	}

	if keepAlive < 0 {
		return time.Duration(math.MaxInt64)
	}

	return keepAlive
}

// LoadTimeout gibt das Timeout fuer Model-Laden zurueck
// Konfigurierbar via OLLAMA_LOAD_TIMEOUT
// 0 oder negative Werte = unendlich
// Default: 5 Minuten
func LoadTimeout() (loadTimeout time.Duration) {
	loadTimeout = 5 * time.Minute
	if s := Var("OLLAMA_LOAD_TIMEOUT"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			loadTimeout = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			loadTimeout = time.Duration(n) * time.Second
		}
	}

	if loadTimeout <= 0 {
		return time.Duration(math.MaxInt64)
	}

	return loadTimeout
}

// Remotes gibt erlaubte Remote-Hosts zurueck
// Konfigurierbar via OLLAMA_REMOTES (komma-separiert)
// Default: ollama.com
func Remotes() []string {
	var r []string
	raw := strings.TrimSpace(Var("OLLAMA_REMOTES"))
	if raw == "" {
		r = []string{"ollama.com"}
	} else {
		r = strings.Split(raw, ",")
	}
	return r
}

// LogLevel gibt das Log-Level zurueck
// Konfigurierbar via OLLAMA_DEBUG
// Werte: 0/false = INFO (Default), 1/true = DEBUG, 2 = TRACE
func LogLevel() slog.Level {
	level := slog.LevelInfo
	if s := Var("OLLAMA_DEBUG"); s != "" {
		if b, _ := strconv.ParseBool(s); b {
			level = slog.LevelDebug
		} else if i, _ := strconv.ParseInt(s, 10, 64); i != 0 {
			level = slog.Level(i * -4)
		}
	}

	return level
}

// Var gibt eine Environment-Variable zurueck
// Entfernt fuehrende/trailing Quotes und Leerzeichen
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}
