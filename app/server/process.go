//go:build windows || darwin

// Modul: process.go
// Beschreibung: Prozess-Management fuer den Ollama Server
// Hauptfunktionen:
//   - cleanup(): Bereinigt vorherige Ollama-Prozesse beim Start
//   - stop(): Wartet auf graceful shutdown eines Prozesses

package server

import (
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

// cleanup prueft die PID-Datei auf einen laufenden Ollama-Prozess
// und faehrt diesen geordnet herunter, falls er laeuft
func cleanup() error {
	data, err := os.ReadFile(pidFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer os.Remove(pidFile)

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return err
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		return nil
	}

	ok, err := terminated(pid)
	if err != nil {
		slog.Debug("cleanup: error checking if terminated", "pid", pid, "err", err)
	}
	if ok {
		return nil
	}

	slog.Info("detected previous ollama process, cleaning up", "pid", pid)
	return stop(proc)
}

// stop wartet auf das Beenden eines Prozesses durch Polling von terminated(pid).
// Falls der Prozess nicht innerhalb von 5 Sekunden beendet wird,
// wird eine Warnung protokolliert und der Prozess gekillt.
func stop(proc *os.Process) error {
	if proc == nil {
		return nil
	}

	if err := terminate(proc); err != nil {
		slog.Warn("graceful terminate failed, killing", "err", err)
		return proc.Kill()
	}

	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()

	for {
		select {
		case <-deadline.C:
			slog.Warn("timeout waiting for graceful shutdown; killing", "pid", proc.Pid)
			return proc.Kill()
		default:
			ok, err := terminated(proc.Pid)
			if err != nil {
				slog.Error("error checking if ollama process is terminated", "err", err)
				return err
			}
			if ok {
				return nil
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
}
