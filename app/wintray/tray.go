//go:build windows

// Package wintray implementiert das Windows System-Tray fuer Ollama.
//
// Modul: tray.go - Kernstrukturen und Initialisierung
// Enthaelt: winTray Struktur, NewTray, InitTray, initInstance
package wintray

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/app/assets"
	"golang.org/x/sys/windows"
)

const (
	UpdateIconName = "tray_upgrade.ico"
	IconName       = "tray.ico"
	ClassName      = "OllamaClass"
)

func NewTray(app AppCallbacks) (TrayCallbacks, error) {
	updateIcon, err := assets.GetIcon(UpdateIconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", UpdateIconName, err)
	}
	icon, err := assets.GetIcon(IconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", IconName, err)
	}

	return InitTray(icon, updateIcon, app)
}

type TrayCallbacks interface {
	Quit()
	TrayRun()
	UpdateAvailable(ver string) error
	GetIconHandle() windows.Handle
}

type AppCallbacks interface {
	UIRun(path string)
	UIShow()
	UITerminate()
	UIRunning() bool
	Quit()
	DoUpdate()
}

type URLSchemeHandler interface {
	HandleURLScheme(urlScheme string)
}

// Helpful sources: https://github.com/golang/exp/blob/master/shiny/driver/internal/win32

// Contains information about loaded resources
type winTray struct {
	instance,
	icon,
	defaultIcon,
	cursor,
	window windows.Handle

	loadedImages   map[string]windows.Handle
	muLoadedImages sync.RWMutex

	// menus keeps track of the submenus keyed by the menu item ID, plus 0
	// which corresponds to the main popup menu.
	menus    map[uint32]windows.Handle
	muMenus  sync.RWMutex
	menuOf   map[uint32]windows.Handle
	muMenuOf sync.RWMutex
	// menuItemIcons maintains the bitmap of each menu item (if applies). It's
	// needed to show the icon correctly when showing a previously hidden menu
	// item again.
	// menuItemIcons   map[uint32]windows.Handle
	// muMenuItemIcons sync.RWMutex
	visibleItems   map[uint32][]uint32
	muVisibleItems sync.RWMutex

	nid   *notifyIconData
	muNID sync.RWMutex
	wcex  *wndClassEx

	wmSystrayMessage,
	wmTaskbarCreated uint32

	pendingUpdate  bool
	updateNotified bool // Only pop up the notification once - TODO consider daily nag?
	normalIcon     []byte
	updateIcon     []byte

	// TODO clean up exit handling
	quitting bool

	app AppCallbacks
}

var wt winTray

func InitTray(icon, updateIcon []byte, app AppCallbacks) (*winTray, error) {
	wt.normalIcon = icon
	wt.updateIcon = updateIcon
	wt.app = app
	if err := wt.initInstance(); err != nil {
		return nil, fmt.Errorf("Unable to init instance: %w\n", err)
	}

	if err := wt.createMenu(); err != nil {
		return nil, fmt.Errorf("Unable to create menu: %w\n", err)
	}

	iconFilePath, err := iconBytesToFilePath(wt.normalIcon)
	if err != nil {
		return nil, fmt.Errorf("Unable to write icon data to temp file: %w", err)
	}
	if err := wt.setIcon(iconFilePath); err != nil {
		return nil, fmt.Errorf("Unable to set icon: %w", err)
	}

	h, err := wt.loadIconFrom(iconFilePath)
	if err != nil {
		return nil, fmt.Errorf("Unable to set default icon: %w", err)
	}
	wt.defaultIcon = h

	return &wt, wt.initMenus()
}

func (t *winTray) initInstance() error {
	const (
		windowName = ""
	)

	t.wmSystrayMessage = WM_USER + 1
	t.visibleItems = make(map[uint32][]uint32)
	t.menus = make(map[uint32]windows.Handle)
	t.menuOf = make(map[uint32]windows.Handle)

	t.loadedImages = make(map[string]windows.Handle)

	taskbarEventNamePtr, _ := windows.UTF16PtrFromString("TaskbarCreated")
	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644947
	res, _, err := pRegisterWindowMessage.Call(
		uintptr(unsafe.Pointer(taskbarEventNamePtr)),
	)
	if res == 0 { // success 0xc000-0xfff
		return fmt.Errorf("failed to register window: %w", err)
	}
	t.wmTaskbarCreated = uint32(res)

	instanceHandle, _, err := pGetModuleHandle.Call(0)
	if instanceHandle == 0 {
		return err
	}
	t.instance = windows.Handle(instanceHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648072(v=vs.85).aspx
	iconHandle, _, err := pLoadIcon.Call(0, uintptr(IDI_APPLICATION))
	if iconHandle == 0 {
		return err
	}
	t.icon = windows.Handle(iconHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648391(v=vs.85).aspx
	cursorHandle, _, err := pLoadCursor.Call(0, uintptr(IDC_ARROW))
	if cursorHandle == 0 {
		return err
	}
	t.cursor = windows.Handle(cursorHandle)

	classNamePtr, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		return err
	}

	windowNamePtr, err := windows.UTF16PtrFromString(windowName)
	if err != nil {
		return err
	}

	t.wcex = &wndClassEx{
		Style:      CS_HREDRAW | CS_VREDRAW,
		WndProc:    windows.NewCallback(t.wndProc),
		Instance:   t.instance,
		Icon:       t.icon,
		Cursor:     t.cursor,
		Background: windows.Handle(6), // (COLOR_WINDOW + 1)
		ClassName:  classNamePtr,
		IconSm:     t.icon,
	}
	if err := t.wcex.register(); err != nil {
		return err
	}

	windowHandle, _, err := pCreateWindowEx.Call(
		uintptr(0),
		uintptr(unsafe.Pointer(classNamePtr)),
		uintptr(unsafe.Pointer(windowNamePtr)),
		uintptr(WS_OVERLAPPEDWINDOW),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(0),
		uintptr(0),
		uintptr(t.instance),
		uintptr(0),
	)
	if windowHandle == 0 {
		return err
	}
	t.window = windows.Handle(windowHandle)

	pShowWindow.Call(uintptr(t.window), uintptr(SW_HIDE)) //nolint:errcheck

	boolRet, _, err := pUpdateWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to update window: %s", err))
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid = &notifyIconData{
		Wnd:             t.window,
		ID:              100,
		Flags:           NIF_MESSAGE,
		CallbackMessage: t.wmSystrayMessage,
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.add()
}

func (t *winTray) GetIconHandle() windows.Handle {
	return t.defaultIcon
}
