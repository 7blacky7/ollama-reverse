// Package updater - Hauptmodul des macOS-Updaters.
// Dieses Modul enthaelt Initialisierung und Basis-Funktionen.
// Upgrade-Logik ist in updater_darwin_upgrade.go,
// Verifizierung in updater_darwin_verify.go.

package updater

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Webkit -framework Cocoa -framework LocalAuthentication -framework ServiceManagement
// #include "updater_darwin.h"
// typedef const char cchar_t;
import "C"

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

var (
	appBackupDir   string
	SystemWidePath = "/Applications/Ollama.app"
)

var BundlePath = func() string {
	if bundle := alreadyMoved(); bundle != "" {
		return bundle
	}

	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	// We also install this binary in Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
	if filepath.Base(exe) == "Squirrel" &&
		filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe)))))) == "Contents" {
		return filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe))))))
	}

	// Make sure we're in a proper macOS app bundle structure (Contents/MacOS)
	if filepath.Base(filepath.Dir(exe)) != "MacOS" ||
		filepath.Base(filepath.Dir(filepath.Dir(exe))) != "Contents" {
		return ""
	}

	return filepath.Dir(filepath.Dir(filepath.Dir(exe)))
}()

func init() {
	VerifyDownload = verifyDownload
	Installer = "Ollama-darwin.zip"
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	var uts unix.Utsname
	if err := unix.Uname(&uts); err == nil {
		sysname := unix.ByteSliceToString(uts.Sysname[:])
		release := unix.ByteSliceToString(uts.Release[:])
		UserAgentOS = fmt.Sprintf("%s/%s", sysname, release)
	} else {
		slog.Warn("unable to determine OS version", "error", err)
		UserAgentOS = "Darwin"
	}

	// TODO handle failure modes here, and developer mode better...

	// Executable = Ollama.app/Contents/MacOS/Ollama

	UpgradeLogFile = filepath.Join(home, ".ollama", "logs", "upgrade.log")

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		slog.Warn("unable to determine user cache dir, falling back to tmpdir", "error", err)
		cacheDir = os.TempDir()
	}
	appDataDir := filepath.Join(cacheDir, "ollama")
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")
	appBackupDir = filepath.Join(appDataDir, "backup")
	UpdateStageDir = filepath.Join(appDataDir, "updates")
}

// DoPostUpgradeCleanup bereinigt nach einem erfolgreichen Upgrade.
func DoPostUpgradeCleanup() error {
	slog.Debug("post upgrade cleanup", "backup", appBackupDir)
	err := os.RemoveAll(appBackupDir)
	if err != nil {
		return err
	}
	slog.Debug("post upgrade cleanup", "old", UpgradeMarkerFile)
	return os.Remove(UpgradeMarkerFile)
}

// DoUpgradeAtStartup fuehrt ein Upgrade beim Starten durch, falls verfuegbar.
func DoUpgradeAtStartup() error {
	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	if BundlePath == "" {
		return fmt.Errorf("unable to upgrade at startup, app in development mode")
	}

	// [Re]verify before proceeding
	if err := VerifyDownload(); err != nil {
		_ = os.Remove(bundle)
		slog.Warn("verification failure", "bundle", bundle, "error", err)
		return nil
	}
	slog.Info("performing update at startup", "bundle", bundle)
	return DoUpgrade(false)
}

// getStagedUpdate gibt den Pfad zum bereitgestellten Update zurueck.
func getStagedUpdate() string {
	files, err := filepath.Glob(filepath.Join(UpdateStageDir, "*", "*.zip"))
	if err != nil {
		slog.Debug("failed to lookup downloads", "error", err)
		return ""
	}
	if len(files) == 0 {
		return ""
	} else if len(files) > 1 {
		// Shouldn't happen
		slog.Warn("multiple update downloads found, using first one", "bundles", files)
	}
	return files[0]
}

// IsUpdatePending prueft, ob ein Update bereitsteht.
func IsUpdatePending() bool {
	return getStagedUpdate() != ""
}

// chownWithAuthorization aendert den Besitzer mit Autorisierung.
func chownWithAuthorization(user string) bool {
	u := C.CString(user)
	defer C.free(unsafe.Pointer(u))
	return (bool)(C.chownWithAuthorization(u))
}

// verifyExtractedBundle verifiziert das entpackte Bundle.
func verifyExtractedBundle(path string) error {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	resp := C.verifyExtractedBundle(p)
	if resp == nil {
		return nil
	}

	return fmt.Errorf("%s", C.GoString(resp))
}

//export goLogInfo
func goLogInfo(msg *C.cchar_t) {
	slog.Info(C.GoString(msg))
}

//export goLogDebug
func goLogDebug(msg *C.cchar_t) {
	slog.Debug(C.GoString(msg))
}

// alreadyMoved prueft, ob die App bereits im Applications-Ordner ist.
func alreadyMoved() string {
	// Respect users intent if they chose "keep" vs. "replace" when dragging to Applications
	installedAppPaths, err := filepath.Glob(filepath.Join(
		strings.TrimSuffix(SystemWidePath, filepath.Ext(SystemWidePath))+"*"+filepath.Ext(SystemWidePath),
		"Contents", "MacOS", "Ollama"))
	if err != nil {
		slog.Warn("failed to lookup installed app paths", "error", err)
		return ""
	}
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("failed to resolve executable", "error", err)
		return ""
	}
	self, err := os.Stat(exe)
	if err != nil {
		slog.Warn("failed to stat running executable", "path", exe, "error", err)
		return ""
	}
	selfSys := self.Sys().(*syscall.Stat_t)
	for _, installedAppPath := range installedAppPaths {
		app, err := os.Stat(installedAppPath)
		if err != nil {
			slog.Debug("failed to stat installed app path", "path", installedAppPath, "error", err)
			continue
		}
		appSys := app.Sys().(*syscall.Stat_t)

		if appSys.Ino == selfSys.Ino {
			return filepath.Dir(filepath.Dir(filepath.Dir(installedAppPath)))
		}
	}
	return ""
}
