// Package readline - Hauptmodul für interaktive Zeileneingabe
//
// Dieses Paket implementiert eine readline-ähnliche Funktionalität mit
// Unterstützung für History, Multiline-Eingabe und Terminal-Raw-Mode.
//
// Hauptkomponenten:
// - Prompt: Konfiguration für Eingabeaufforderungen
// - Instance: Hauptinstanz für readline-Operationen
// - New: Konstruktor für neue Readline-Instanzen
// - Readline: Hauptmethode zum Lesen einer Zeile

package readline

import (
	"fmt"
	"io"
	"os"
)

// Prompt definiert die Eingabeaufforderungs-Konfiguration
type Prompt struct {
	Prompt         string
	AltPrompt      string
	Placeholder    string
	AltPlaceholder string
	UseAlt         bool
}

// prompt gibt den aktuellen Prompt-String zurück
func (p *Prompt) prompt() string {
	if p.UseAlt {
		return p.AltPrompt
	}
	return p.Prompt
}

// placeholder gibt den aktuellen Placeholder-String zurück
func (p *Prompt) placeholder() string {
	if p.UseAlt {
		return p.AltPlaceholder
	}
	return p.Placeholder
}

// Instance ist die Hauptstruktur für readline-Operationen
type Instance struct {
	Prompt      *Prompt
	Terminal    *Terminal
	History     *History
	Pasting     bool
	pastedLines []string
}

// New erstellt eine neue Readline-Instanz mit dem angegebenen Prompt
func New(prompt Prompt) (*Instance, error) {
	term, err := NewTerminal()
	if err != nil {
		return nil, err
	}

	history, err := NewHistory()
	if err != nil {
		return nil, err
	}

	return &Instance{
		Prompt:   &prompt,
		Terminal: term,
		History:  history,
	}, nil
}

// Readline liest eine Zeile vom Terminal mit Unterstützung für
// History-Navigation, Cursor-Bewegung und Multiline-Eingabe
func (i *Instance) Readline() (string, error) {
	if !i.Terminal.rawmode {
		fd := os.Stdin.Fd()
		termios, err := SetRawMode(fd)
		if err != nil {
			return "", err
		}
		i.Terminal.rawmode = true
		i.Terminal.termios = termios
	}

	prompt := i.Prompt.prompt()
	if i.Pasting {
		// Bei Paste immer Alt-Prompt verwenden
		prompt = i.Prompt.AltPrompt
	}
	fmt.Print(prompt)

	defer func() {
		fd := os.Stdin.Fd()
		//nolint:errcheck
		UnsetRawMode(fd, i.Terminal.termios)
		i.Terminal.rawmode = false
	}()

	buf, _ := NewBuffer(i.Prompt)

	var esc bool
	var escex bool
	var metaDel bool
	var currentLineBuf []rune

	// draining verfolgt ob wir gepufferten Input aus dem Cooked-Mode verarbeiten.
	// Im Cooked-Mode sendet Enter \n, aber im Raw-Mode sendet Ctrl+J \n.
	// Wir behandeln \n aus dem Cooked-Mode als Submit, nicht als Multiline.
	var draining, stopDraining bool

	for {
		// Verzögerte Zustandsänderung aus vorheriger Iteration anwenden
		if stopDraining {
			draining = false
			stopDraining = false
		}

		// Placeholder nur anzeigen wenn nicht im Paste-Modus (außer bei Multiline)
		showPlaceholder := !i.Pasting || i.Prompt.UseAlt
		if buf.IsEmpty() && showPlaceholder {
			ph := i.Prompt.placeholder()
			fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
		}

		r, err := i.Terminal.Read()

		// Nach dem Lesen prüfen ob mehr gepufferte Daten vorhanden sind
		if i.Terminal.reader.Buffered() > 0 {
			draining = true
		} else if draining {
			stopDraining = true
		}

		if buf.IsEmpty() {
			fmt.Print(ClearToEOL)
		}

		if err != nil {
			return "", io.EOF
		}

		// Escape-Ex-Sequenzen verarbeiten
		if escex {
			shouldContinue, err := i.processEscapeEx(r, buf, &currentLineBuf, &escex, &metaDel)
			if err != nil {
				return "", err
			}
			if shouldContinue {
				continue
			}
		} else if esc {
			if i.processEscape(r, buf, &esc, &escex) {
				continue
			}
		}

		// Normale Zeichen verarbeiten
		output, done, err := i.processCharacter(r, buf, &currentLineBuf, &esc, &metaDel, draining)
		if done {
			return output, err
		}
	}
}

// HistoryEnable aktiviert die History-Funktionalität
func (i *Instance) HistoryEnable() {
	i.History.Enabled = true
}

// HistoryDisable deaktiviert die History-Funktionalität
func (i *Instance) HistoryDisable() {
	i.History.Enabled = false
}

// historyPrev navigiert zur vorherigen History-Eintrag
func (i *Instance) historyPrev(buf *Buffer, currentLineBuf *[]rune) {
	if i.History.Pos > 0 {
		if i.History.Pos == i.History.Size() {
			*currentLineBuf = []rune(buf.String())
		}
		buf.Replace([]rune(i.History.Prev()))
	}
}

// historyNext navigiert zum nächsten History-Eintrag
func (i *Instance) historyNext(buf *Buffer, currentLineBuf *[]rune) {
	if i.History.Pos < i.History.Size() {
		buf.Replace([]rune(i.History.Next()))
		if i.History.Pos == i.History.Size() {
			buf.Replace(*currentLineBuf)
		}
	}
}
