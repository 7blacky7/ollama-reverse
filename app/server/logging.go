//go:build windows || darwin

// Modul: logging.go
// Beschreibung: Log-Rotation fuer den Ollama Server
// Hauptfunktionen:
//   - openRotatingLog(): Oeffnet oder erstellt die rotierende Log-Datei

package server

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/app/logrotate"
)

// openRotatingLog oeffnet eine rotierende Log-Datei.
// Das Verzeichnis wird erstellt, falls es nicht existiert.
// Die Log-Rotation wird bei jedem Aufruf durchgefuehrt.
func openRotatingLog() (io.WriteCloser, error) {
	// TODO: Rotation basierend auf Groesse oder Zeit erwaegen
	dir := filepath.Dir(serverLogPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log directory: %w", err)
	}

	logrotate.Rotate(serverLogPath)
	f, err := os.OpenFile(serverLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open log file: %w", err)
	}
	return f, nil
}
