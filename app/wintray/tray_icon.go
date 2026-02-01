//go:build windows

// Package wintray implementiert das Windows System-Tray fuer Ollama.
//
// Modul: tray_icon.go - Icon-Funktionen
// Enthaelt: setIcon, loadIconFrom, iconBytesToFilePath, DisplayFirstUseNotification
package wintray

import (
	"crypto/md5"
	"encoding/hex"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func iconBytesToFilePath(iconBytes []byte) (string, error) {
	bh := md5.Sum(iconBytes)
	dataHash := hex.EncodeToString(bh[:])
	iconFilePath := filepath.Join(os.TempDir(), "ollama_temp_icon_"+dataHash)

	if _, err := os.Stat(iconFilePath); os.IsNotExist(err) {
		if err := os.WriteFile(iconFilePath, iconBytes, 0o644); err != nil {
			return "", err
		}
	}
	return iconFilePath, nil
}

// Loads an image from file and shows it in tray.
// Shell_NotifyIcon: https://msdn.microsoft.com/en-us/library/windows/desktop/bb762159(v=vs.85).aspx
func (t *winTray) setIcon(src string) error {
	h, err := t.loadIconFrom(src)
	if err != nil {
		return err
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid.Icon = h
	t.nid.Flags |= NIF_ICON | NIF_TIP
	if toolTipUTF16, err := syscall.UTF16FromString("Ollama"); err == nil {
		copy(t.nid.Tip[:], toolTipUTF16)
	} else {
		return err
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.modify()
}

// Loads an image from file to be shown in tray or menu item.
// LoadImage: https://msdn.microsoft.com/en-us/library/windows/desktop/ms648045(v=vs.85).aspx
func (t *winTray) loadIconFrom(src string) (windows.Handle, error) {
	// Save and reuse handles of loaded images
	t.muLoadedImages.RLock()
	h, ok := t.loadedImages[src]
	t.muLoadedImages.RUnlock()
	if !ok {
		srcPtr, err := windows.UTF16PtrFromString(src)
		if err != nil {
			return 0, err
		}
		res, _, err := pLoadImage.Call(
			0,
			uintptr(unsafe.Pointer(srcPtr)),
			IMAGE_ICON,
			0,
			0,
			LR_LOADFROMFILE|LR_DEFAULTSIZE,
		)
		if res == 0 {
			return 0, err
		}
		h = windows.Handle(res)
		t.muLoadedImages.Lock()
		t.loadedImages[src] = h
		t.muLoadedImages.Unlock()
	}
	return h, nil
}

func (t *winTray) DisplayFirstUseNotification() error {
	t.muNID.Lock()
	defer t.muNID.Unlock()
	copy(t.nid.InfoTitle[:], windows.StringToUTF16(firstTimeTitle))
	copy(t.nid.Info[:], windows.StringToUTF16(firstTimeMessage))
	t.nid.Flags |= NIF_INFO
	t.nid.Size = uint32(unsafe.Sizeof(*wt.nid))

	return t.nid.modify()
}
