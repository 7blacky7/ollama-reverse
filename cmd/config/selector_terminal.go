// Modul: selector_terminal.go
// Beschreibung: Terminal I/O Handling für das Selector-UI.
// Enthält Raw-Mode-Steuerung, Input-Parsing und Hilfsfunktionen.

package config

import (
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

// Terminal I/O handling

type terminalState struct {
	fd       int
	oldState *term.State
}

func enterRawMode() (*terminalState, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return nil, err
	}
	fmt.Fprint(os.Stderr, ansiHideCursor)
	return &terminalState{fd: fd, oldState: oldState}, nil
}

func (t *terminalState) restore() {
	fmt.Fprint(os.Stderr, ansiShowCursor)
	term.Restore(t.fd, t.oldState)
}

func clearLines(n int) {
	if n > 0 {
		fmt.Fprintf(os.Stderr, "\033[%dA", n)
		fmt.Fprint(os.Stderr, ansiClearDown)
	}
}

func parseInput(r io.Reader) (inputEvent, byte, error) {
	buf := make([]byte, 3)
	n, err := r.Read(buf)
	if err != nil {
		return 0, 0, err
	}

	switch {
	case n == 1 && buf[0] == 13:
		return eventEnter, 0, nil
	case n == 1 && (buf[0] == 3 || buf[0] == 27):
		return eventEscape, 0, nil
	case n == 1 && buf[0] == 9:
		return eventTab, 0, nil
	case n == 1 && buf[0] == 127:
		return eventBackspace, 0, nil
	case n == 3 && buf[0] == 27 && buf[1] == 91 && buf[2] == 65:
		return eventUp, 0, nil
	case n == 3 && buf[0] == 27 && buf[1] == 91 && buf[2] == 66:
		return eventDown, 0, nil
	case n == 1 && buf[0] >= 32 && buf[0] < 127:
		return eventChar, buf[0], nil
	}

	return eventNone, 0, nil
}

func filterItems(items []selectItem, filter string) []selectItem {
	if filter == "" {
		return items
	}
	var result []selectItem
	filterLower := strings.ToLower(filter)
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}
