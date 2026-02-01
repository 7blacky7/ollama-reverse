//go:build windows || darwin

// Package main - URL-Schema-Handler und Browser-Funktionen.
// Dieses Modul verarbeitet ollama:// URLs und oeffnet Links im Browser.

package main

import (
	"fmt"
	"log/slog"
	"net/url"
	"os/exec"
	"runtime"
	"strings"

	"github.com/ollama/ollama/app/auth"
)

// handleConnectURLScheme holt die Connect-URL und oeffnet sie im Browser.
func handleConnectURLScheme() {
	if checkUserLoggedIn(uiServerPort) {
		slog.Info("user is already logged in, opening app instead")
		showWindow(wv.webview.Window())
		return
	}

	connectURL, err := auth.BuildConnectURL("https://ollama.com")
	if err != nil {
		slog.Error("failed to build connect URL", "error", err)
		openInBrowser("https://ollama.com/connect")
		return
	}

	openInBrowser(connectURL)
}

// openInBrowser oeffnet die angegebene URL im Standard-Browser.
func openInBrowser(url string) {
	var cmd string
	var args []string

	switch runtime.GOOS {
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler", url}
	case "darwin":
		cmd = "open"
		args = []string{url}
	default: // "linux", "freebsd", "openbsd", "netbsd"... should not reach here
		slog.Warn("unsupported OS for openInBrowser", "os", runtime.GOOS)
	}

	slog.Info("executing browser command", "cmd", cmd, "args", args)
	if err := exec.Command(cmd, args...).Start(); err != nil {
		slog.Error("failed to open URL in browser", "url", url, "cmd", cmd, "args", args, "error", err)
	}
}

// parseURLScheme parst eine ollama:// URL und validiert sie.
// Unterstuetzt: ollama:// (App oeffnen) und ollama://connect (OAuth).
func parseURLScheme(urlSchemeRequest string) (isConnect bool, err error) {
	parsedURL, err := url.Parse(urlSchemeRequest)
	if err != nil {
		return false, fmt.Errorf("invalid URL: %w", err)
	}

	// Check if this is a connect URL
	if parsedURL.Host == "connect" || strings.TrimPrefix(parsedURL.Path, "/") == "connect" {
		return true, nil
	}

	// Allow bare ollama:// or ollama:/// to open the app
	if (parsedURL.Host == "" && parsedURL.Path == "") || parsedURL.Path == "/" {
		return false, nil
	}

	return false, fmt.Errorf("unsupported ollama:// URL path: %s", urlSchemeRequest)
}

// handleURLSchemeInCurrentInstance verarbeitet URL-Schema-Anfragen in der aktuellen Instanz.
func handleURLSchemeInCurrentInstance(urlSchemeRequest string) {
	isConnect, err := parseURLScheme(urlSchemeRequest)
	if err != nil {
		slog.Error("failed to parse URL scheme request", "url", urlSchemeRequest, "error", err)
		return
	}

	if isConnect {
		handleConnectURLScheme()
	} else {
		if wv.webview != nil {
			showWindow(wv.webview.Window())
		}
	}
}
