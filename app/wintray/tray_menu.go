//go:build windows

// Package wintray implementiert das Windows System-Tray fuer Ollama.
//
// Modul: tray_menu.go - Menu-Funktionen
// Enthaelt: createMenu, addOrUpdateMenuItem, addSeparatorMenuItem, showMenu, VisibleItems-Verwaltung
package wintray

import (
	"fmt"
	"log/slog"
	"sort"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Contains information about a menu item.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms647578(v=vs.85).aspx
type menuItemInfo struct {
	Size, Mask, Type, State     uint32
	ID                          uint32
	SubMenu, Checked, Unchecked windows.Handle
	ItemData                    uintptr
	TypeData                    *uint16
	Cch                         uint32
	BMPItem                     windows.Handle
}

func (t *winTray) createMenu() error {
	menuHandle, _, err := pCreatePopupMenu.Call()
	if menuHandle == 0 {
		return err
	}
	t.menus[0] = windows.Handle(menuHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms647575(v=vs.85).aspx
	mi := struct {
		Size, Mask, Style, Max uint32
		Background             windows.Handle
		ContextHelpID          uint32
		MenuData               uintptr
	}{
		Mask: MIM_APPLYTOSUBMENUS,
	}
	mi.Size = uint32(unsafe.Sizeof(mi))

	res, _, err := pSetMenuInfo.Call(
		uintptr(t.menus[0]),
		uintptr(unsafe.Pointer(&mi)),
	)
	if res == 0 {
		return err
	}
	return nil
}

func (t *winTray) addOrUpdateMenuItem(menuItemId uint32, parentId uint32, title string, disabled bool) error {
	titlePtr, err := windows.UTF16PtrFromString(title)
	if err != nil {
		return err
	}

	mi := menuItemInfo{
		Mask:     MIIM_FTYPE | MIIM_STRING | MIIM_ID | MIIM_STATE,
		Type:     MFT_STRING,
		ID:       menuItemId,
		TypeData: titlePtr,
		Cch:      uint32(len(title)),
	}
	mi.Size = uint32(unsafe.Sizeof(mi))
	if disabled {
		mi.State |= MFS_DISABLED
	}

	var res uintptr
	t.muMenus.RLock()
	menu := t.menus[parentId]
	t.muMenus.RUnlock()
	if t.getVisibleItemIndex(parentId, menuItemId) != -1 {
		// We set the menu item info based on the menuID
		boolRet, _, err := pSetMenuItemInfo.Call(
			uintptr(menu),
			uintptr(menuItemId),
			0,
			uintptr(unsafe.Pointer(&mi)),
		)
		if boolRet == 0 {
			return fmt.Errorf("failed to set menu item: %w", err)
		}
	}

	if res == 0 {
		// Menu item does not already exist, create it
		t.muMenus.RLock()
		submenu, exists := t.menus[menuItemId]
		t.muMenus.RUnlock()
		if exists {
			mi.Mask |= MIIM_SUBMENU
			mi.SubMenu = submenu
		}
		t.addToVisibleItems(parentId, menuItemId)
		position := t.getVisibleItemIndex(parentId, menuItemId)
		res, _, err = pInsertMenuItem.Call(
			uintptr(menu),
			uintptr(position),
			1,
			uintptr(unsafe.Pointer(&mi)),
		)
		if res == 0 {
			t.delFromVisibleItems(parentId, menuItemId)
			return err
		}
		t.muMenuOf.Lock()
		t.menuOf[menuItemId] = menu
		t.muMenuOf.Unlock()
	}

	return nil
}

func (t *winTray) addSeparatorMenuItem(menuItemId, parentId uint32) error {
	mi := menuItemInfo{
		Mask: MIIM_FTYPE | MIIM_ID | MIIM_STATE,
		Type: MFT_SEPARATOR,
		ID:   menuItemId,
	}

	mi.Size = uint32(unsafe.Sizeof(mi))

	t.addToVisibleItems(parentId, menuItemId)
	position := t.getVisibleItemIndex(parentId, menuItemId)
	t.muMenus.RLock()
	menu := uintptr(t.menus[parentId])
	t.muMenus.RUnlock()
	res, _, err := pInsertMenuItem.Call(
		menu,
		uintptr(position),
		1,
		uintptr(unsafe.Pointer(&mi)),
	)
	if res == 0 {
		return err
	}

	return nil
}

// func (t *winTray) hideMenuItem(menuItemId, parentId uint32) error {
// 	const ERROR_SUCCESS syscall.Errno = 0

// 	t.muMenus.RLock()
// 	menu := uintptr(t.menus[parentId])
// 	t.muMenus.RUnlock()
// 	res, _, err := pRemoveMenu.Call(
// 		menu,
// 		uintptr(menuItemId),
// 		MF_BYCOMMAND,
// 	)
// 	if res == 0 && err.(syscall.Errno) != ERROR_SUCCESS {
// 		return err
// 	}
// 	t.delFromVisibleItems(parentId, menuItemId)

// 	return nil
// }

func (t *winTray) showMenu() error {
	p := point{}
	boolRet, _, err := pGetCursorPos.Call(uintptr(unsafe.Pointer(&p)))
	if boolRet == 0 {
		return err
	}
	boolRet, _, err = pSetForegroundWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Warn(fmt.Sprintf("failed to bring menu to foreground: %s", err))
	}

	boolRet, _, err = pTrackPopupMenu.Call(
		uintptr(t.menus[0]),
		TPM_BOTTOMALIGN|TPM_LEFTALIGN|TPM_RIGHTBUTTON,
		uintptr(p.X),
		uintptr(p.Y),
		0,
		uintptr(t.window),
		0,
	)
	if boolRet == 0 {
		return err
	}

	return nil
}

func (t *winTray) delFromVisibleItems(parent, val uint32) {
	t.muVisibleItems.Lock()
	defer t.muVisibleItems.Unlock()
	visibleItems := t.visibleItems[parent]
	for i, itemval := range visibleItems {
		if val == itemval {
			t.visibleItems[parent] = append(visibleItems[:i], visibleItems[i+1:]...)
			break
		}
	}
}

func (t *winTray) addToVisibleItems(parent, val uint32) {
	t.muVisibleItems.Lock()
	defer t.muVisibleItems.Unlock()
	if visibleItems, exists := t.visibleItems[parent]; !exists {
		t.visibleItems[parent] = []uint32{val}
	} else {
		newvisible := append(visibleItems, val)
		sort.Slice(newvisible, func(i, j int) bool { return newvisible[i] < newvisible[j] })
		t.visibleItems[parent] = newvisible
	}
}

func (t *winTray) getVisibleItemIndex(parent, val uint32) int {
	t.muVisibleItems.RLock()
	defer t.muVisibleItems.RUnlock()
	for i, itemval := range t.visibleItems[parent] {
		if val == itemval {
			return i
		}
	}
	return -1
}
