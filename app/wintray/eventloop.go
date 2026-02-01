//go:build windows

// Package wintray - Event-Loop und Window-Procedure.
//
// Diese Datei enthaelt:
// - TrayRun: Haupt-Message-Pump
// - wndProc: Windows-Callback fuer Message-Processing
// - quit: Beendet die Anwendung
// - SendUIRequestMessage: Sendet UI-Anfragen
package wintray

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

// ============================================================================
// Konstanten
// ============================================================================

var (
	quitOnce            sync.Once
	UI_REQUEST_MSG_ID   = WM_USER + 2
	FOCUS_WINDOW_MSG_ID = WM_USER + 3
)

// ============================================================================
// TrayRun - Haupt-Event-Loop
// ============================================================================

// TrayRun startet die Haupt-Message-Pump.
func (t *winTray) TrayRun() {
	slog.Debug("starting event handling loop")
	m := &struct {
		WindowHandle windows.Handle
		Message      uint32
		Wparam       uintptr
		Lparam       uintptr
		Time         uint32
		Pt           point
		LPrivate     uint32
	}{}
	for {
		ret, _, err := pGetMessage.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0)

		// WM_QUIT Messages vom UI-Fenster ignorieren, die nicht die Haupt-App beenden sollen
		if m.Message == WM_QUIT && t.app.UIRunning() {
			if t.app != nil {
				slog.Debug("converting WM_QUIT to terminate call on webview")
				t.app.UITerminate()
			}
			// Weitere WM_QUIT Messages abarbeiten
			for {
				ret, _, err = pGetMessage.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0)
				if m.Message != WM_QUIT {
					break
				}
			}
		}

		// Rueckgabewerte:
		// - Nicht WM_QUIT: Rueckgabe ist nicht-null
		// - WM_QUIT: Rueckgabe ist null
		// - Fehler: Rueckgabe ist -1
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644936(v=vs.85).aspx
		switch int32(ret) {
		case -1:
			slog.Error(fmt.Sprintf("get message failure: %v", err))
			return
		case 0:
			return
		default:
			pTranslateMessage.Call(uintptr(unsafe.Pointer(m))) //nolint:errcheck
			pDispatchMessage.Call(uintptr(unsafe.Pointer(m)))  //nolint:errcheck
		}
	}
}

// ============================================================================
// wndProc - Window Procedure Callback
// ============================================================================

// wndProc verarbeitet Messages, die an das Fenster gesendet werden.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms633573(v=vs.85).aspx
func (t *winTray) wndProc(hWnd windows.Handle, message uint32, wParam, lParam uintptr) (lResult uintptr) {
	switch message {
	case WM_COMMAND:
		lResult = t.handleCommand(hWnd, message, wParam, lParam)
	case WM_CLOSE:
		t.handleClose()
	case WM_DESTROY:
		defer pPostQuitMessage.Call(uintptr(int32(0))) //nolint:errcheck
		fallthrough
	case WM_ENDSESSION:
		t.handleEndSession()
	case t.wmSystrayMessage:
		lResult = t.handleSystrayMessage(hWnd, message, wParam, lParam)
	case t.wmTaskbarCreated:
		t.handleTaskbarCreated()
	case uint32(UI_REQUEST_MSG_ID):
		t.handleUIRequest(wParam, lParam)
	case WM_COPYDATA:
		lResult = t.handleCopyData(lParam)
	case uint32(FOCUS_WINDOW_MSG_ID):
		lResult = t.handleFocusWindow()
	default:
		// Standard-Verarbeitung fuer nicht behandelte Messages
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms633572(v=vs.85).aspx
		lResult, _, _ = pDefWindowProc.Call(
			uintptr(hWnd),
			uintptr(message),
			wParam,
			lParam,
		)
	}
	return
}

// handleCommand verarbeitet WM_COMMAND Messages (Menu-Aktionen).
func (t *winTray) handleCommand(hWnd windows.Handle, message uint32, wParam, lParam uintptr) uintptr {
	menuItemId := int32(wParam)
	switch menuItemId {
	case quitMenuID:
		t.app.Quit()
	case updateMenuID:
		t.app.DoUpdate()
	case openUIMenuID:
		t.app.UIShow()
	case settingsUIMenuID:
		t.app.UIRun("/settings")
	case diagLogsMenuID:
		t.showLogs()
	default:
		slog.Debug(fmt.Sprintf("Unexpected menu item id: %d", menuItemId))
		lResult, _, _ := pDefWindowProc.Call(
			uintptr(hWnd),
			uintptr(message),
			wParam,
			lParam,
		)
		return lResult
	}
	return 0
}

// handleClose verarbeitet WM_CLOSE.
func (t *winTray) handleClose() {
	boolRet, _, err := pDestroyWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to destroy window: %s", err))
	}
	err = t.wcex.unregister()
	if err != nil {
		slog.Error(fmt.Sprintf("failed to unregister window %s", err))
	}
}

// handleEndSession verarbeitet WM_ENDSESSION und WM_DESTROY.
func (t *winTray) handleEndSession() {
	t.muNID.Lock()
	if t.nid != nil {
		err := t.nid.delete()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to delete nid: %s", err))
		}
	}
	t.muNID.Unlock()
}

// handleSystrayMessage verarbeitet Systray-Messages.
func (t *winTray) handleSystrayMessage(hWnd windows.Handle, message uint32, wParam, lParam uintptr) uintptr {
	switch lParam {
	case WM_MOUSEMOVE, WM_LBUTTONDOWN:
		// Ignorieren
	case WM_RBUTTONUP, WM_LBUTTONUP:
		err := t.showMenu()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to show menu: %s", err))
		}
	case 0x405: // Notification Left-Click
		if t.pendingUpdate {
			t.app.DoUpdate()
		}
	case 0x404: // Middle-Click oder Close-Notification
		// Nichts tun
	default:
		slog.Debug(fmt.Sprintf("unmanaged app message, lParm: 0x%x", lParam))
		lResult, _, _ := pDefWindowProc.Call(
			uintptr(hWnd),
			uintptr(message),
			wParam,
			lParam,
		)
		return lResult
	}
	return 0
}

// handleTaskbarCreated behandelt explorer.exe Neustarts.
func (t *winTray) handleTaskbarCreated() {
	t.muNID.Lock()
	err := t.nid.add()
	if err != nil {
		slog.Error(fmt.Sprintf("failed to refresh the taskbar on explorer restart: %s", err))
	}
	t.muNID.Unlock()
}

// handleUIRequest verarbeitet UI-Anfragen.
func (t *winTray) handleUIRequest(wParam, lParam uintptr) {
	l := int(wParam)
	path := unsafe.String((*byte)(unsafe.Pointer(lParam)), l) //nolint:govet,gosec
	t.app.UIRun(path)
}

// handleCopyData verarbeitet WM_COPYDATA (URL-Schema von anderen Instanzen).
func (t *winTray) handleCopyData(lParam uintptr) uintptr {
	if lParam != 0 {
		cds := (*COPYDATASTRUCT)(unsafe.Pointer(lParam)) //nolint:govet,gosec
		if cds.DwData == 1 {                             // Identifier fuer URL-Schema-Messages
			data := make([]byte, cds.CbData)
			copy(data, (*[1 << 30]byte)(unsafe.Pointer(cds.LpData))[:cds.CbData:cds.CbData]) //nolint:govet,gosec
			urlScheme := string(data)
			handleURLSchemeRequest(urlScheme)
			return 1 // Erfolg
		}
	}
	return 0
}

// handleFocusWindow verarbeitet Focus-Anfragen von anderen Instanzen.
func (t *winTray) handleFocusWindow() uintptr {
	if t.app.UIRunning() {
		t.app.UIShow()
	} else {
		t.app.UIRun("/")
	}
	return 1 // Erfolg
}

// ============================================================================
// Quit und SendUIRequestMessage
// ============================================================================

// Quit beendet die Tray-Anwendung.
func (t *winTray) Quit() {
	t.quitting = true
	quitOnce.Do(quit)
}

// SendUIRequestMessage sendet eine UI-Anfrage an das Tray-Fenster.
func SendUIRequestMessage(path string) {
	boolRet, _, err := pPostMessage.Call(
		uintptr(wt.window),
		uintptr(UI_REQUEST_MSG_ID),
		uintptr(len(path)),
		uintptr(unsafe.Pointer(unsafe.StringData(path))),
	)
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to post UI request message %s", err))
	}
}

// quit sendet WM_CLOSE an das Tray-Fenster.
func quit() {
	boolRet, _, err := pPostMessage.Call(
		uintptr(wt.window),
		WM_CLOSE,
		0,
		0,
	)
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to post close message on shutdown %s", err))
	}
}
