//go:build windows || darwin

// Webview-Bindings-Modul: JavaScript-Bindings fuer Dateioperationen und UI
// Dieses Modul enthaelt alle JavaScript-Bindings fuer die WebView-Instanz.
// Aufgeteilt aus der urspruenglichen webview.go (528 LOC)
// Siehe auch: webview_styles.go

package main

// #include "menu.h"
import "C"

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/app/dialog"
	"github.com/ollama/ollama/app/webview"
)

// initWebviewBindings initialisiert alle JavaScript-Bindings
func initWebviewBindings(wv webview.WebView, w *Webview) {
	// Zoom bindings
	wv.Bind("zoomIn", func() {
		current := wv.GetZoom()
		wv.SetZoom(current + 0.1)
	})

	wv.Bind("zoomOut", func() {
		current := wv.GetZoom()
		wv.SetZoom(current - 0.1)
	})

	wv.Bind("zoomReset", func() {
		wv.SetZoom(1.0)
	})

	wv.Bind("ready", func() {
		showWindow(wv.Window())
	})

	wv.Bind("close", func() {
		hideWindow(wv.Window())
	})

	// File system bindings
	bindSelectModelsDirectory(wv)
	bindSelectFiles(wv)
	bindSelectWorkingDirectory(wv)

	// Window bindings
	wv.Bind("drag", func() {
		wv.Dispatch(func() {
			drag(wv.Window())
		})
	})

	wv.Bind("doubleClick", func() {
		wv.Dispatch(func() {
			doubleClick(wv.Window())
		})
	})

	// Context menu binding
	bindContextMenu(wv)

	// Resize binding with debounce
	bindResize(wv, w)
}

// bindSelectModelsDirectory bindet die Verzeichnisauswahl fuer Models
func bindSelectModelsDirectory(wv webview.WebView) {
	wv.Bind("selectModelsDirectory", func() {
		go func() {
			callCallback := func(data interface{}) {
				dataJSON, _ := json.Marshal(data)
				wv.Dispatch(func() {
					wv.Eval(fmt.Sprintf("window.__selectModelsDirectoryCallback && window.__selectModelsDirectoryCallback(%s)", dataJSON))
				})
			}

			directory, err := dialog.Directory().Title("Select Model Directory").ShowHidden(true).Browse()
			if err != nil {
				slog.Debug("Directory selection cancelled or failed", "error", err)
				callCallback(nil)
				return
			}
			slog.Debug("Directory selected", "path", directory)
			callCallback(directory)
		}()
	})
}

// bindSelectFiles bindet die Mehrfach-Dateiauswahl
func bindSelectFiles(wv webview.WebView) {
	wv.Bind("selectFiles", func() {
		go func() {
			callCallback := func(data interface{}) {
				dataJSON, _ := json.Marshal(data)
				wv.Dispatch(func() {
					wv.Eval(fmt.Sprintf("window.__selectFilesCallback && window.__selectFilesCallback(%s)", dataJSON))
				})
			}

			// Define allowed extensions for native dialog filtering
			textExts := []string{
				"pdf", "docx", "txt", "md", "csv", "json", "xml", "html", "htm",
				"js", "jsx", "ts", "tsx", "py", "java", "cpp", "c", "cc", "h", "cs", "php", "rb",
				"go", "rs", "swift", "kt", "scala", "sh", "bat", "yaml", "yml", "toml", "ini",
				"cfg", "conf", "log", "rtf",
			}
			imageExts := []string{"png", "jpg", "jpeg", "webp"}
			allowedExts := append(textExts, imageExts...)

			filenames, err := dialog.File().
				Filter("Supported Files", allowedExts...).
				Title("Select Files").
				LoadMultiple()
			if err != nil {
				slog.Debug("Multiple file selection cancelled or failed", "error", err)
				callCallback(nil)
				return
			}

			if len(filenames) == 0 {
				callCallback(nil)
				return
			}

			files := processSelectedFiles(filenames, allowedExts)
			if len(files) == 0 {
				callCallback(nil)
			} else {
				callCallback(files)
			}
		}()
	})
}

// processSelectedFiles verarbeitet die ausgewaehlten Dateien
func processSelectedFiles(filenames []string, allowedExts []string) []map[string]string {
	var files []map[string]string
	maxFileSize := int64(10 * 1024 * 1024) // 10MB

	for _, filename := range filenames {
		ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filename), "."))
		validExt := false
		for _, allowedExt := range allowedExts {
			if ext == allowedExt {
				validExt = true
				break
			}
		}
		if !validExt {
			slog.Warn("file extension not allowed, skipping", "filename", filepath.Base(filename), "extension", ext)
			continue
		}

		fileStat, err := os.Stat(filename)
		if err != nil {
			slog.Error("failed to get file info", "error", err, "filename", filename)
			continue
		}

		if fileStat.Size() > maxFileSize {
			slog.Warn("file too large, skipping", "filename", filepath.Base(filename), "size", fileStat.Size())
			continue
		}

		fileBytes, err := os.ReadFile(filename)
		if err != nil {
			slog.Error("failed to read file", "error", err, "filename", filename)
			continue
		}

		mimeType := http.DetectContentType(fileBytes)
		dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(fileBytes))

		fileResult := map[string]string{
			"filename": filepath.Base(filename),
			"path":     filename,
			"dataURL":  dataURL,
		}

		files = append(files, fileResult)
	}

	return files
}

// bindSelectWorkingDirectory bindet die Arbeitsverzeichnis-Auswahl
func bindSelectWorkingDirectory(wv webview.WebView) {
	wv.Bind("selectWorkingDirectory", func() {
		go func() {
			callCallback := func(data interface{}) {
				dataJSON, _ := json.Marshal(data)
				wv.Dispatch(func() {
					wv.Eval(fmt.Sprintf("window.__selectWorkingDirectoryCallback && window.__selectWorkingDirectoryCallback(%s)", dataJSON))
				})
			}

			directory, err := dialog.Directory().Title("Select Working Directory").ShowHidden(true).Browse()
			if err != nil {
				slog.Debug("Directory selection cancelled or failed", "error", err)
				callCallback(nil)
				return
			}
			slog.Debug("Directory selected", "path", directory)
			callCallback(directory)
		}()
	})
}

// bindContextMenu bindet die Kontextmenue-Funktionalitaet
func bindContextMenu(wv webview.WebView) {
	wv.Bind("setContextMenuItems", func(items []map[string]interface{}) error {
		menuMutex.Lock()
		defer menuMutex.Unlock()

		if len(menuItems) > 0 {
			pinner.Unpin()
		}

		menuItems = nil
		for _, item := range items {
			menuItem := C.menuItem{
				label:     C.CString(item["label"].(string)),
				enabled:   0,
				separator: 0,
			}

			if item["enabled"] != nil {
				menuItem.enabled = 1
			}

			if item["separator"] != nil {
				menuItem.separator = 1
			}
			menuItems = append(menuItems, menuItem)
		}
		return nil
	})
}

// bindResize bindet die Fenstergroessen-Aenderung mit Debounce
func bindResize(wv webview.WebView, w *Webview) {
	var resizeTimer *time.Timer
	var resizeMutex sync.Mutex

	wv.Bind("resize", func(width, height int) {
		if w.Store != nil {
			resizeMutex.Lock()
			if resizeTimer != nil {
				resizeTimer.Stop()
			}
			resizeTimer = time.AfterFunc(100*time.Millisecond, func() {
				err := w.Store.SetWindowSize(width, height)
				if err != nil {
					slog.Error("failed to set window size", "error", err)
				}
			})
			resizeMutex.Unlock()
		}
	})
}
