// Modul: selector_types.go
// Beschreibung: Typen, Konstanten und Basisstrukturen für das Selector-UI.
// Enthält ANSI-Escape-Sequenzen, Event-Typen und Item-Strukturen.

package config

import "errors"

// ANSI escape sequences for terminal formatting.
const (
	ansiHideCursor = "\033[?25l"
	ansiShowCursor = "\033[?25h"
	ansiBold       = "\033[1m"
	ansiReset      = "\033[0m"
	ansiGray       = "\033[37m"
	ansiClearDown  = "\033[J"
)

const maxDisplayedItems = 10

var errCancelled = errors.New("cancelled")

type selectItem struct {
	Name        string
	Description string
}

type inputEvent int

const (
	eventNone inputEvent = iota
	eventEnter
	eventEscape
	eventUp
	eventDown
	eventTab
	eventBackspace
	eventChar
)
