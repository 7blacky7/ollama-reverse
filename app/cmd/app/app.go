//go:build windows || darwin

// Package main - Hauptmodul der Ollama Desktop-Anwendung.
// Dieses Modul enthaelt die main-Funktion und Kommandozeilen-Verarbeitung.
// Weitere Funktionen sind in app_handlers.go, app_tasks.go und app_init.go.

package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/app/logrotate"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
)

var (
	wv           = &Webview{}
	uiServerPort int
)

var debug = strings.EqualFold(os.Getenv("OLLAMA_DEBUG"), "true") || os.Getenv("OLLAMA_DEBUG") == "1"

var (
	fastStartup = false
	devMode     = false
)

type appMove int

const (
	CannotMove appMove = iota
	UserDeclinedMove
	MoveCompleted
	AlreadyMoved
	LoginSession
	PermissionDenied
	MoveError
)

// parseArgs verarbeitet Kommandozeilen-Argumente.
func parseArgs() (startHidden bool, urlSchemeRequest string) {
	if len(os.Args) <= 1 {
		return false, ""
	}

	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "ollama://") {
			urlSchemeRequest = arg
			slog.Info("received URL scheme request", "url", arg)
			continue
		}
		switch arg {
		case "serve":
			fmt.Fprintln(os.Stderr, "serve command not supported, use ollama")
			os.Exit(1)
		case "version", "-v", "--version":
			fmt.Println(version.Version)
			os.Exit(0)
		case "background":
			fmt.Fprintln(os.Stdout, "starting in background")
			runInBackground()
			os.Exit(0)
		case "hidden", "-j", "--hide":
			startHidden = true
		case "--fast-startup":
			fastStartup = true
		case "-dev", "--dev":
			devMode = true
		}
	}
	return startHidden, urlSchemeRequest
}

func main() {
	startHidden, urlSchemeRequest := parseArgs()

	level := slog.LevelInfo
	if debug {
		level = slog.LevelDebug
	}

	logrotate.Rotate(appLogPath)
	if _, err := setupLogging(level); err != nil {
		slog.Error(err.Error())
		return
	}
	logStartup()

	// On Windows, check if another instance is running and send URL to it
	if runtime.GOOS == "windows" && urlSchemeRequest != "" {
		slog.Debug("checking for existing instance", "url", urlSchemeRequest)
		if checkAndHandleExistingInstance(urlSchemeRequest) {
			// Successfully sent to another instance
		} else {
			go func() { handleURLSchemeInCurrentInstance(urlSchemeRequest) }()
		}
	}

	if u := os.Getenv("OLLAMA_UPDATE_URL"); u != "" {
		updater.UpdateCheckURLBase = u
	}

	// Detect if this is a first start after an upgrade
	var skipMove bool
	if _, err := os.Stat(updater.UpgradeMarkerFile); err == nil {
		slog.Debug("first start after upgrade")
		if err := updater.DoPostUpgradeCleanup(); err != nil {
			slog.Error("failed to cleanup prior version", "error", err)
		}
		skipMove = true
		startHidden = true
	}

	if !skipMove && !fastStartup {
		if maybeMoveAndRestart() == MoveCompleted {
			return
		}
	}

	handleExistingInstance(startHidden)
	installSymlink()

	// Setup network listener
	var ln net.Listener
	var err error
	if devMode {
		ln, err = net.Listen("tcp", "127.0.0.1:3001")
	} else {
		ln, err = net.Listen("tcp", "127.0.0.1:0")
	}
	if err != nil {
		slog.Error("failed to find available port", "error", err)
		return
	}

	port := ln.Addr().(*net.TCPAddr).Port
	token := uuid.NewString()
	wv.port = port
	wv.token = token
	uiServerPort = port

	st := &store.Store{}

	// Enable CORS in development mode
	if devMode {
		os.Setenv("OLLAMA_CORS", "1")
		if err := checkViteDevServer(); err != nil {
			slog.Error(err.Error())
			fmt.Fprintln(os.Stderr, "Error: Vite dev server is not running on port 5173")
			fmt.Fprintln(os.Stderr, "Please run 'npm run dev' in the ui/app directory")
			os.Exit(1)
		}
	}

	toolRegistry := tools.NewRegistry()
	slog.Info("initialized tools registry", "tool_count", len(toolRegistry.List()))

	ctx, cancel := context.WithCancel(context.Background())

	srv, uiServer, done := startServers(&serverConfig{
		ctx:          ctx,
		cancel:       cancel,
		store:        st,
		token:        token,
		toolRegistry: toolRegistry,
		ln:           ln,
	})

	upd := &updater.Updater{Store: st}
	upd.StartBackgroundUpdaterChecker(ctx, UpdateAvailable)

	hasCompletedFirstRun, err := st.HasCompletedFirstRun()
	if err != nil {
		slog.Error("failed to load has completed first run", "error", err)
	}

	if !hasCompletedFirstRun {
		if err := st.SetHasCompletedFirstRun(true); err != nil {
			slog.Error("failed to set has completed first run", "error", err)
		}
	}

	// capture SIGINT and SIGTERM signals
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		slog.Info("received SIGINT or SIGTERM signal, shutting down")
		quit()
	}()

	if urlSchemeRequest != "" {
		go func() { handleURLSchemeInCurrentInstance(urlSchemeRequest) }()
	} else {
		slog.Debug("no URL scheme request to handle")
	}

	go func() {
		slog.Debug("waiting for ollama server to be ready")
		if err := ui.WaitForServer(ctx, 10*time.Second); err != nil {
			slog.Warn("ollama server not ready, continuing anyway", "error", err)
		}
		if _, err := uiServer.UserData(ctx); err != nil {
			slog.Warn("failed to load user data", "error", err)
		}
	}()

	osRun(cancel, hasCompletedFirstRun, startHidden)

	slog.Info("shutting down desktop server")
	if err := srv.Close(); err != nil {
		slog.Warn("error shutting down desktop server", "error", err)
	}

	slog.Info("shutting down ollama server")
	cancel()
	<-done
}
