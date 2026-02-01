// Package llm - Runner Subprocess Verwaltung
//
// Funktionen zum Starten und Konfigurieren des Runner-Subprocesses:
// - StartRunner: Hauptfunktion zum Starten des Runners
// - findAvailablePort: Freien Port finden
// - configureRunnerEnv: Umgebungsvariablen setzen
// - setupRunnerOutput: Stdout/Stderr weiterleiten
package llm

import (
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// StartRunner startet den Runner-Subprocess
func StartRunner(ollamaEngine bool, modelPath string, gpuLibs []string, out io.Writer, extraEnvs map[string]string) (cmd *exec.Cmd, port int, err error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, 0, fmt.Errorf("unable to lookup executable path: %w", err)
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	port = findAvailablePort()

	params := buildRunnerParams(ollamaEngine, modelPath, port)
	cmd = exec.Command(exe, params...)
	cmd.Env = os.Environ()

	if out != nil {
		if err := setupRunnerOutput(cmd, out); err != nil {
			return nil, 0, err
		}
	}
	cmd.SysProcAttr = LlamaServerSysProcAttr

	configureRunnerEnv(cmd, gpuLibs, extraEnvs)

	slog.Info("starting runner", "cmd", cmd)
	slog.Debug("subprocess", "", filteredEnv(cmd.Env))

	if err = cmd.Start(); err != nil {
		return nil, 0, err
	}

	return cmd, port, nil
}

// findAvailablePort findet einen freien TCP Port
func findAvailablePort() int {
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port := l.Addr().(*net.TCPAddr).Port
			l.Close()
			return port
		}
	}
	slog.Debug("ResolveTCPAddr failed, using random port")
	return rand.Intn(65535-49152) + 49152
}

// buildRunnerParams erstellt die Kommandozeilenparameter
func buildRunnerParams(ollamaEngine bool, modelPath string, port int) []string {
	params := []string{"runner"}
	if ollamaEngine {
		params = append(params, "--ollama-engine")
	}
	if modelPath != "" {
		params = append(params, "--model", modelPath)
	}
	params = append(params, "--port", strconv.Itoa(port))
	return params
}

// setupRunnerOutput verbindet Stdout/Stderr mit dem Writer
func setupRunnerOutput(cmd *exec.Cmd, out io.Writer) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to spawn server stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to spawn server stderr pipe: %w", err)
	}
	go func() { io.Copy(out, stdout) }()
	go func() { io.Copy(out, stderr) }()
	return nil
}

// configureRunnerEnv konfiguriert die Umgebungsvariablen für den Runner
func configureRunnerEnv(cmd *exec.Cmd, gpuLibs []string, extraEnvs map[string]string) {
	pathEnv := getPathEnvName()
	libraryPaths := append([]string{}, gpuLibs...)
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}

	pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))
	updateEnvVars(cmd, pathEnv, pathEnvVal, gpuLibs, extraEnvs)
}

// getPathEnvName gibt den plattformspezifischen Library-Path Namen zurück
func getPathEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}

// updateEnvVars aktualisiert die Umgebungsvariablen im Cmd
func updateEnvVars(cmd *exec.Cmd, pathEnv, pathEnvVal string, gpuLibs []string, extraEnvs map[string]string) {
	pathNeeded := true
	ollamaPathNeeded := true
	extraEnvsDone := make(map[string]bool)
	for k := range extraEnvs {
		extraEnvsDone[k] = false
	}

	for i := range cmd.Env {
		cmp := strings.SplitN(cmd.Env[i], "=", 2)
		if strings.EqualFold(cmp[0], pathEnv) {
			cmd.Env[i] = pathEnv + "=" + pathEnvVal
			pathNeeded = false
		} else if strings.EqualFold(cmp[0], "OLLAMA_LIBRARY_PATH") {
			cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, string(filepath.ListSeparator))
			ollamaPathNeeded = false
		} else if len(extraEnvs) != 0 {
			for k, v := range extraEnvs {
				if strings.EqualFold(cmp[0], k) {
					cmd.Env[i] = k + "=" + v
					extraEnvsDone[k] = true
				}
			}
		}
	}

	if pathNeeded {
		cmd.Env = append(cmd.Env, pathEnv+"="+pathEnvVal)
	}
	if ollamaPathNeeded {
		cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(gpuLibs, string(filepath.ListSeparator)))
	}
	for k, done := range extraEnvsDone {
		if !done {
			cmd.Env = append(cmd.Env, k+"="+extraEnvs[k])
		}
	}
}
