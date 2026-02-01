//go:build windows || darwin

// app_windows_init.go - Windows App Initialisierung und Konfiguration
//
// Enthaelt:
// - Package-Variablen und Konstanten
// - init(): Pfad-Initialisierung
// - maybeMoveAndRestart, handleExistingInstance, installSymlink
// - appCallbacks Struktur und Methoden
package main

import (
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/app/wintray"
	"golang.org/x/sys/windows"
)

var (
	u32                  = windows.NewLazySystemDLL("User32.dll")
	pBringWindowToTop    = u32.NewProc("BringWindowToTop")
	pShowWindow          = u32.NewProc("ShowWindow")
	pSendMessage         = u32.NewProc("SendMessageA")
	pGetSystemMetrics    = u32.NewProc("GetSystemMetrics")
	pGetWindowRect       = u32.NewProc("GetWindowRect")
	pSetWindowPos        = u32.NewProc("SetWindowPos")
	pSetForegroundWindow = u32.NewProc("SetForegroundWindow")
	pSetActiveWindow     = u32.NewProc("SetActiveWindow")
	pIsIconic            = u32.NewProc("IsIconic")

	appPath         = filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", "Ollama")
	appLogPath      = filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "app.log")
	startupShortcut = filepath.Join(os.Getenv("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "Ollama.lnk")
	ollamaPath      string
	DesktopAppName  = "ollama app.exe"
)

func init() {
	// With alternate install location use executable location
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("error discovering executable directory", "error", err)
	} else {
		appPath = filepath.Dir(exe)
	}
	ollamaPath = filepath.Join(appPath, "ollama.exe")

	// Handle developer mode (go run ./cmd/app)
	if _, err := os.Stat(ollamaPath); err != nil {
		pwd, err := os.Getwd()
		if err != nil {
			slog.Warn("missing ollama.exe and failed to get pwd", "error", err)
			return
		}
		distAppPath := filepath.Join(pwd, "dist", "windows-"+runtime.GOARCH)
		distOllamaPath := filepath.Join(distAppPath, "ollama.exe")
		if _, err := os.Stat(distOllamaPath); err == nil {
			slog.Info("detected developer mode")
			appPath = distAppPath
			ollamaPath = distOllamaPath
		}
	}
}

func maybeMoveAndRestart() appMove {
	return 0
}

// handleExistingInstance checks for existing instances and optionally focuses them
func handleExistingInstance(startHidden bool) {
	if wintray.CheckAndFocusExistingInstance(!startHidden) {
		slog.Info("existing instance found, exiting")
		os.Exit(0)
	}
}

func installSymlink() {}

type appCallbacks struct {
	t        wintray.TrayCallbacks
	shutdown func()
}

var app = &appCallbacks{}

func (ac *appCallbacks) UIRun(path string) {
	wv.Run(path)
}

func (*appCallbacks) UIShow() {
	if wv.webview != nil {
		showWindow(wv.webview.Window())
	} else {
		wv.Run("/")
	}
}

func (*appCallbacks) UITerminate() {
	wv.Terminate()
}

func (*appCallbacks) UIRunning() bool {
	return wv.IsRunning()
}

func (app *appCallbacks) Quit() {
	app.t.Quit()
	wv.Terminate()
}

// TODO - reconcile with above for consistency between mac/windows
func quit() {
	wv.Terminate()
}

func (app *appCallbacks) DoUpdate() {
	// Safeguard in case we have requests in flight that need to drain...
	slog.Info("Waiting for server to shutdown")

	app.shutdown()

	if err := updater.DoUpgrade(true); err != nil {
		slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
	}
}

// HandleURLScheme implements the URLSchemeHandler interface
func (app *appCallbacks) HandleURLScheme(urlScheme string) {
	handleURLSchemeRequest(urlScheme)
}

// handleURLSchemeRequest processes URL scheme requests from other instances
func handleURLSchemeRequest(urlScheme string) {
	isConnect, err := parseURLScheme(urlScheme)
	if err != nil {
		slog.Error("failed to parse URL scheme request", "url", urlScheme, "error", err)
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

func UpdateAvailable(ver string) error {
	return app.t.UpdateAvailable(ver)
}

func logStartup() {
	slog.Info("starting Ollama", "app", appPath, "version", version.Version, "OS", updater.UserAgentOS)
}

func LaunchNewApp() {
}

func osRun(shutdown func(), hasCompletedFirstRun, startHidden bool) {
	var err error
	app.shutdown = shutdown
	app.t, err = wintray.NewTray(app)
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	// TODO - can this be generalized?
	go func() {
		<-signals
		slog.Debug("shutting down due to signal")
		app.t.Quit()
		wv.Terminate()
	}()

	// On windows, we run the final tasks in the main thread
	// before starting the tray event loop.  These final tasks
	// may trigger the UI, and must do that from the main thread.
	if !startHidden {
		// Determine if the process was started from a shortcut
		// ~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\Ollama
		const STARTF_TITLEISLINKNAME = 0x00000800
		var info windows.StartupInfo
		if err := windows.GetStartupInfo(&info); err != nil {
			slog.Debug("unable to retrieve startup info", "error", err)
		} else if info.Flags&STARTF_TITLEISLINKNAME == STARTF_TITLEISLINKNAME {
			linkPath := windows.UTF16PtrToString(info.Title)
			if strings.Contains(linkPath, "Startup") {
				startHidden = true
			}
		}
	}
	if startHidden {
		startHiddenTasks()
	} else {
		ptr := wv.Run("/")

		// Set the window icon using the tray icon
		if ptr != nil {
			iconHandle := app.t.GetIconHandle()
			if iconHandle != 0 {
				hwnd := uintptr(ptr)
				const ICON_SMALL = 0
				const ICON_BIG = 1
				const WM_SETICON = 0x0080

				pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_SMALL), uintptr(iconHandle))
				pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_BIG), uintptr(iconHandle))
			}
		}

		centerWindow(ptr)
	}

	if !hasCompletedFirstRun {
		// Only create the login shortcut on first start
		// so we can respect users deletion of the link
		err = createLoginShortcut()
		if err != nil {
			slog.Warn("unable to create login shortcut", "error", err)
		}
	}

	app.t.TrayRun() // This will block the main thread
}
