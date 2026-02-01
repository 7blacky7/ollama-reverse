//go:build windows || darwin

// Modul: path.go
// Beschreibung: Pfad-Aufloesung fuer die Ollama-Binaries
// Hauptfunktionen:
//   - resolvePath(): Findet den Pfad zur Ollama-Executable

package server

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// resolvePath sucht nach dem angegebenen Binary in folgender Reihenfolge:
// 1. Im App-Bundle (macOS) oder Anwendungsverzeichnis (Windows)
// 2. Im Entwicklungs-dist-Verzeichnis
// 3. Im System-PATH
func resolvePath(name string) string {
	// Zuerst im App-Bundle suchen
	if exe, _ := os.Executable(); exe != "" {
		var dir string
		if runtime.GOOS == "windows" {
			dir = filepath.Dir(exe)
		} else {
			dir = filepath.Join(filepath.Dir(exe), "..", "Resources")
		}
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return filepath.Join(dir, name)
		}
	}

	// Entwicklungs-dist-Pfad pruefen
	for _, path := range []string{
		filepath.Join("dist", runtime.GOOS, name),
		filepath.Join("dist", runtime.GOOS+"-"+runtime.GOARCH, name),
	} {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// Fallback auf System-PATH
	if p, _ := exec.LookPath(name); p != "" {
		return p
	}

	return name
}
