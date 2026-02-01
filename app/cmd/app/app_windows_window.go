//go:build windows || darwin

// app_windows_window.go - Windows Fenster-Management
//
// Enthaelt:
// - Window-Konstanten (SW_*, SM_*, SWP_*, MF_*)
// - POINT und Rect Strukturen
// - centerWindow: Fenster zentrieren
// - showWindow: Fenster anzeigen und fokussieren
// - hideWindow: Fenster verstecken
// - runInBackground: Hintergrund-Ausfuehrung
// - drag, doubleClick: Event-Handler
// - checkAndHandleExistingInstance: Instanz-Check
package main

import (
	"log/slog"
	"os"
	"os/exec"
	"unsafe"

	"github.com/ollama/ollama/app/wintray"
)

const (
	SW_HIDE        = 0  // Hides the window
	SW_SHOW        = 5  // Shows window in its current size/position
	SW_SHOWNA      = 8  // Shows without activating
	SW_MINIMIZE    = 6  // Minimizes the window
	SW_RESTORE     = 9  // Restores to previous size/position
	SW_SHOWDEFAULT = 10 // Sets show state based on program state
	SM_CXSCREEN    = 0
	SM_CYSCREEN    = 1
	HWND_TOP       = 0
	SWP_NOSIZE     = 0x0001
	SWP_NOMOVE     = 0x0002
	SWP_NOZORDER   = 0x0004
	SWP_SHOWWINDOW = 0x0040

	// Menu constants
	MF_STRING     = 0x00000000
	MF_SEPARATOR  = 0x00000800
	MF_GRAYED     = 0x00000001
	TPM_RETURNCMD = 0x0100
)

// POINT structure for cursor position
type POINT struct {
	X int32
	Y int32
}

// Rect structure for GetWindowRect
type Rect struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

// centerWindow zentriert das Fenster auf dem Bildschirm
func centerWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd == 0 {
		return
	}

	var rect Rect
	pGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&rect)))

	screenWidth, _, _ := pGetSystemMetrics.Call(uintptr(SM_CXSCREEN))
	screenHeight, _, _ := pGetSystemMetrics.Call(uintptr(SM_CYSCREEN))

	windowWidth := rect.Right - rect.Left
	windowHeight := rect.Bottom - rect.Top

	x := (int32(screenWidth) - windowWidth) / 2
	y := (int32(screenHeight) - windowHeight) / 2

	// Ensure the window is not positioned off-screen
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}

	pSetWindowPos.Call(
		hwnd,
		uintptr(HWND_TOP),
		uintptr(x),
		uintptr(y),
		uintptr(windowWidth),  // Keep original width
		uintptr(windowHeight), // Keep original height
		uintptr(SWP_SHOWWINDOW),
	)
}

// showWindow zeigt das Fenster an und bringt es in den Vordergrund
func showWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd != 0 {
		iconHandle := app.t.GetIconHandle()
		if iconHandle != 0 {
			const ICON_SMALL = 0
			const ICON_BIG = 1
			const WM_SETICON = 0x0080

			pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_SMALL), uintptr(iconHandle))
			pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_BIG), uintptr(iconHandle))
		}

		// Check if window is minimized
		isMinimized, _, _ := pIsIconic.Call(hwnd)
		if isMinimized != 0 {
			// Restore the window if it's minimized
			pShowWindow.Call(hwnd, uintptr(SW_RESTORE))
		}

		// Show the window
		pShowWindow.Call(hwnd, uintptr(SW_SHOW))

		// Bring window to top
		pBringWindowToTop.Call(hwnd)

		// Force window to foreground
		pSetForegroundWindow.Call(hwnd)

		// Make it the active window
		pSetActiveWindow.Call(hwnd)

		// Ensure window is positioned on top
		pSetWindowPos.Call(
			hwnd,
			uintptr(HWND_TOP),
			0, 0, 0, 0,
			uintptr(SWP_NOSIZE|SWP_NOMOVE|SWP_SHOWWINDOW),
		)
	}
}

// hideWindow versteckt das Anwendungsfenster
func hideWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd != 0 {
		pShowWindow.Call(
			hwnd,
			uintptr(SW_HIDE),
		)
	}
}

// runInBackground startet die Anwendung im Hintergrund
func runInBackground() {
	exe, err := os.Executable()
	if err != nil {
		slog.Error("failed to get executable path", "error", err)
		os.Exit(1)
	}
	cmd := exec.Command(exe, "hidden")
	if cmd != nil {
		err = cmd.Run()
		if err != nil {
			slog.Error("failed to run Ollama", "exe", exe, "error", err)
			os.Exit(1)
		}
	} else {
		slog.Error("failed to start Ollama", "exe", exe)
		os.Exit(1)
	}
}

func drag(ptr unsafe.Pointer) {}

func doubleClick(ptr unsafe.Pointer) {}

// checkAndHandleExistingInstance prueft ob eine andere Instanz laeuft und sendet die URL
func checkAndHandleExistingInstance(urlSchemeRequest string) bool {
	if urlSchemeRequest == "" {
		return false
	}

	// Try to send URL to existing instance using wintray messaging
	if wintray.CheckAndSendToExistingInstance(urlSchemeRequest) {
		os.Exit(0)
		return true
	}

	// No existing instance, we'll handle it ourselves
	return false
}
