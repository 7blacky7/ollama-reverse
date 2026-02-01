// server.go - Server-Struktur und Lifecycle-Management fuer Image-Generation
// Dieses Modul enthaelt die Hauptstruktur des Servers sowie Start-, Stop- und Health-Check-Logik.
package imagegen

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// Server wraps an image generation subprocess to implement llm.LlamaServer.
//
// This implementation is compatible with Ollama's scheduler and can be loaded/unloaded
// like any other model. The plan is to eventually bring this into the llm/ package
// and evolve llm/ to support MLX and multimodal models. For now, keeping the code
// separate allows for independent iteration on image generation support.
type Server struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelName   string
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string // Last stderr line for error reporting
	lastErrLock sync.Mutex
}

// NewServer spawns a new image generation subprocess and waits until it's ready.
func NewServer(modelName string) (*Server, error) {
	// Validate platform support before attempting to start
	if err := CheckPlatformSupport(); err != nil {
		return nil, err
	}

	// Find a free port
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}

	// Get the current executable path (we use the same binary with runner subcommand)
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --image-engine --model <path> --port <port>
	cmd := exec.Command(exe, "runner", "--image-engine", "--model", modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// On Linux, set LD_LIBRARY_PATH to include MLX library directories
	if runtime.GOOS == "linux" {
		// Build library paths: start with LibOllamaPath, then add any mlx_* subdirectories
		libraryPaths := []string{ml.LibOllamaPath}
		if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_*")); err == nil {
			libraryPaths = append(libraryPaths, mlxDirs...)
		}

		// Append existing LD_LIBRARY_PATH if set
		if existingPath, ok := os.LookupEnv("LD_LIBRARY_PATH"); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(existingPath)...)
		}

		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		// Update or add LD_LIBRARY_PATH in cmd.Env
		found := false
		for i := range cmd.Env {
			if strings.HasPrefix(cmd.Env[i], "LD_LIBRARY_PATH=") {
				cmd.Env[i] = "LD_LIBRARY_PATH=" + pathEnvVal
				found = true
				break
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+pathEnvVal)
		}
		slog.Debug("mlx subprocess library path", "LD_LIBRARY_PATH", pathEnvVal)
	}

	// Get total weight size from manifest
	var weightSize uint64
	if manifest, err := LoadManifest(modelName); err == nil {
		weightSize = uint64(manifest.TotalTensorSize())
	}

	s := &Server{
		cmd:       cmd,
		port:      port,
		modelName: modelName,
		vramSize:  weightSize,
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
	}

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			slog.Info("image-runner", "msg", scanner.Text())
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			slog.Warn("image-runner", "msg", line)
			s.lastErrLock.Lock()
			s.lastErr = line
			s.lastErrLock.Unlock()
		}
	}()

	slog.Info("starting image runner subprocess", "exe", exe, "model", modelName, "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start image runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		err := cmd.Wait()
		s.done <- err
	}()

	// Wait for subprocess to be ready
	if err := s.waitUntilRunning(); err != nil {
		s.Close()
		return nil, err
	}

	return s, nil
}

// ModelPath returns the path to the model.
func (s *Server) ModelPath() string {
	return s.modelName
}

// Load implements the LlamaServer interface for model loading.
func (s *Server) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	return nil, nil
}

// Ping checks if the subprocess is healthy.
func (s *Server) Ping(ctx context.Context) error {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.port)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

// waitUntilRunning waits for the subprocess to be ready.
func (s *Server) waitUntilRunning() error {
	ctx := context.Background()
	timeout := time.After(2 * time.Minute)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-s.done:
			// Include recent stderr lines for better error context
			errMsg := s.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("image runner failed: %s (exit: %v)", errMsg, err)
			}
			return fmt.Errorf("image runner exited unexpectedly: %w", err)
		case <-timeout:
			errMsg := s.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("timeout waiting for image runner: %s", errMsg)
			}
			return errors.New("timeout waiting for image runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("image runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// getLastErr returns the last stderr line.
func (s *Server) getLastErr() string {
	s.lastErrLock.Lock()
	defer s.lastErrLock.Unlock()
	return s.lastErr
}

// WaitUntilRunning implements the LlamaServer interface.
func (s *Server) WaitUntilRunning(ctx context.Context) error { return nil }

// Close terminates the subprocess.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != nil && s.cmd.Process != nil {
		slog.Info("stopping image runner subprocess", "pid", s.cmd.Process.Pid)
		s.cmd.Process.Signal(os.Interrupt)

		// Wait briefly for graceful shutdown
		select {
		case <-s.done:
		case <-time.After(5 * time.Second):
			s.cmd.Process.Kill()
		}
		s.cmd = nil
	}
	return nil
}

// Pid returns the process ID of the subprocess.
func (s *Server) Pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

// GetPort returns the port the subprocess is listening on.
func (s *Server) GetPort() int { return s.port }

// HasExited returns true if the subprocess has exited.
func (s *Server) HasExited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}
