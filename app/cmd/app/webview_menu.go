//go:build windows || darwin

// Webview-Menu-Modul: Kontextmenue-Verwaltung und C-Interop
// Dieses Modul verwaltet die nativen Kontextmenue-Funktionen.
// Aufgeteilt aus der urspruenglichen webview.go (528 LOC)

package main

// #include "menu.h"
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

var (
	menuItems []C.menuItem
	menuMutex sync.RWMutex
	pinner    runtime.Pinner
)

// wv ist eine globale Referenz fuer das Kontextmenue
var wv *Webview

// SetWebviewReference setzt die globale Webview-Referenz fuer das Menue
func SetWebviewReference(w *Webview) {
	wv = w
}

//export menu_get_item_count
func menu_get_item_count() C.int {
	menuMutex.RLock()
	defer menuMutex.RUnlock()
	return C.int(len(menuItems))
}

//export menu_get_items
func menu_get_items() unsafe.Pointer {
	menuMutex.RLock()
	defer menuMutex.RUnlock()

	if len(menuItems) == 0 {
		return nil
	}

	// Return pointer to the slice data
	pinner.Pin(&menuItems[0])
	return unsafe.Pointer(&menuItems[0])
}

//export menu_handle_selection
func menu_handle_selection(item *C.char) {
	if wv != nil && wv.webview != nil {
		wv.webview.Eval(fmt.Sprintf("window.handleContextMenuResult('%s')", C.GoString(item)))
	}
}
