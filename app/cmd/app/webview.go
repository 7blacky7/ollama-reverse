//go:build windows || darwin

// Webview-Modul: Hauptstruktur und Initialisierung
// Dieses Modul verwaltet die WebView-Instanz fuer die Desktop-Anwendung.
// Aufgeteilt aus der urspruenglichen webview.go (528 LOC)
// Siehe auch: webview_bindings.go, webview_menu.go

package main

// #include "menu.h"
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/webview"
)

type Webview struct {
	port    int
	token   string
	webview webview.WebView
	mutex   sync.Mutex

	Store *store.Store
}

// Run initializes the webview and starts its event loop.
// Note: this must be called from the primary app thread
// This returns the OS native window handle to the caller
func (w *Webview) Run(path string) unsafe.Pointer {
	var url string
	if devMode {
		// In development mode, use the local dev server
		url = fmt.Sprintf("http://localhost:5173%s", path)
	} else {
		url = fmt.Sprintf("http://127.0.0.1:%d%s", w.port, path)
	}
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.webview == nil {
		// Note: turning on debug on macos throws errors but is marginally functional for debugging
		// TODO (jmorganca): we should pre-create the window and then provide it here to
		// webview so we can hide it from the start and make other modifications
		wv := webview.New(debug)
		// start the window hidden
		hideWindow(wv.Window())
		wv.SetTitle("Ollama")

		// Initialize JavaScript and bindings
		initWebviewJS(wv, w)
		initWebviewBindings(wv, w)

		// On Darwin, we can't have 2 threads both running global event loops
		// but on Windows, the event loops are tied to the window, so we're
		// able to run in both the tray and webview
		if runtime.GOOS != "darwin" {
			go func() {
				wv.Run()
			}()
		}

		if w.Store != nil {
			width, height, err := w.Store.WindowSize()
			if err != nil {
				// Fehler beim Laden der Fenstergroesse
			}
			if width > 0 && height > 0 {
				wv.SetSize(width, height, webview.HintNone)
			} else {
				wv.SetSize(800, 600, webview.HintNone)
			}
		}
		wv.SetSize(800, 600, webview.HintMin)

		w.webview = wv
		w.webview.Navigate(url)
	} else {
		w.webview.Eval(fmt.Sprintf(`
			history.pushState({}, '', '%s');
		`, path))
		showWindow(w.webview.Window())
	}

	return w.webview.Window()
}

func (w *Webview) Terminate() {
	w.mutex.Lock()
	if w.webview == nil {
		w.mutex.Unlock()
		return
	}

	wv := w.webview
	w.webview = nil
	w.mutex.Unlock()
	wv.Terminate()
	wv.Destroy()
}

func (w *Webview) IsRunning() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	return w.webview != nil
}
