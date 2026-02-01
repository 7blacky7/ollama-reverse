//go:build windows

// Package wintray - Instanz-Erkennung und Inter-Process-Communication.
//
// Diese Datei enthaelt:
// - findExistingInstance: Sucht nach laufender Ollama-Instanz
// - CheckAndSendToExistingInstance: Sendet URL-Schema an bestehende Instanz
// - CheckAndFocusExistingInstance: Fokussiert bestehende Instanz
// - handleURLSchemeRequest: Verarbeitet URL-Schema-Anfragen
package wintray

import (
	"log/slog"
	"unsafe"

	"golang.org/x/sys/windows"
)

// ============================================================================
// Instanz-Suche
// ============================================================================

// findExistingInstance sucht nach einem existierenden Ollama-Instanz-Fenster.
// Gibt den Window-Handle zurueck wenn gefunden, sonst 0.
func findExistingInstance() uintptr {
	classNamePtr, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		slog.Error("failed to convert class name to UTF16", "error", err)
		return 0
	}

	hwnd, _, _ := pFindWindow.Call(
		uintptr(unsafe.Pointer(classNamePtr)),
		0, // Window-Name (null = beliebig)
	)

	return hwnd
}

// ============================================================================
// URL-Schema Handling
// ============================================================================

// CheckAndSendToExistingInstance versucht ein URL-Schema an eine existierende
// Instanz zu senden.
// Gibt true zurueck wenn erfolgreich gesendet, false wenn keine Instanz gefunden.
func CheckAndSendToExistingInstance(urlScheme string) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		// Kein existierendes Fenster gefunden
		return false
	}

	data := []byte(urlScheme)
	cds := COPYDATASTRUCT{
		DwData: 1, // 1 identifiziert URL-Schema-Messages
		CbData: uint32(len(data)),
		LpData: uintptr(unsafe.Pointer(&data[0])),
	}

	result, _, err := pSendMessage.Call(
		hwnd,
		uintptr(WM_COPYDATA),
		0, // wParam ist Handle zum sendenden Fenster (0 ist ok)
		uintptr(unsafe.Pointer(&cds)),
	)

	// SendMessage gibt das Ergebnis der Window-Procedure zurueck
	// Fuer WM_COPYDATA bedeutet nicht-null Erfolg
	if result == 0 {
		slog.Error("failed to send URL scheme message to existing instance", "error", err)
		return false
	}
	return true
}

// handleURLSchemeRequest verarbeitet eine URL-Schema-Anfrage.
func handleURLSchemeRequest(urlScheme string) {
	if urlScheme == "" {
		slog.Warn("empty URL scheme request")
		return
	}

	// App-Callback aufrufen fuer URL-Schema-Verarbeitung
	if wt.app != nil {
		if urlHandler, ok := wt.app.(URLSchemeHandler); ok {
			urlHandler.HandleURLScheme(urlScheme)
		} else {
			slog.Warn("app does not implement URLSchemeHandler interface")
		}
	} else {
		slog.Warn("wt.app is nil")
	}
}

// ============================================================================
// Focus-Handling
// ============================================================================

// CheckAndFocusExistingInstance versucht eine existierende Instanz zu finden
// und optional zu fokussieren.
// Gibt true zurueck wenn eine existierende Instanz gefunden wurde, sonst false.
func CheckAndFocusExistingInstance(shouldFocus bool) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		// Kein existierendes Fenster gefunden
		return false
	}

	if !shouldFocus {
		slog.Info("existing instance found, not focusing due to startHidden")
		return true
	}

	// Focus-Message an existierende Instanz senden
	result, _, err := pSendMessage.Call(
		hwnd,
		uintptr(FOCUS_WINDOW_MSG_ID),
		0, // wParam nicht verwendet
		0, // lParam nicht verwendet
	)

	// SendMessage gibt das Ergebnis der Window-Procedure zurueck
	// Fuer unsere Custom-Message bedeutet nicht-null Erfolg
	if result == 0 {
		slog.Error("failed to send focus message to existing instance", "error", err)
		return false
	}

	slog.Info("sent focus request to existing instance")

	return true
}
