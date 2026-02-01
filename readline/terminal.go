// Package readline - Terminal-Modul
//
// Dieses Modul enthält die Terminal-Strukturen und -Methoden für die
// Raw-Mode-Terminal-Interaktion.
//
// Hauptkomponenten:
// - Terminal: Struktur für Terminal-I/O mit Buffered Reader
// - NewTerminal: Konstruktor für Terminal-Instanzen
// - Read: Liest einzelne Runes vom Terminal

package readline

import (
	"bufio"
	"os"
)

// Terminal verwaltet die Terminal-Ein-/Ausgabe im Raw-Mode
type Terminal struct {
	reader  *bufio.Reader
	rawmode bool
	termios any
}

// NewTerminal erstellt eine neue Terminal-Instanz und testet den Raw-Mode
func NewTerminal() (*Terminal, error) {
	fd := os.Stdin.Fd()
	termios, err := SetRawMode(fd)
	if err != nil {
		return nil, err
	}
	if err := UnsetRawMode(fd, termios); err != nil {
		return nil, err
	}

	t := &Terminal{
		reader: bufio.NewReader(os.Stdin),
	}

	return t, nil
}

// Read liest ein einzelnes Rune vom Terminal
func (t *Terminal) Read() (rune, error) {
	r, _, err := t.reader.ReadRune()
	if err != nil {
		return 0, err
	}
	return r, nil
}
