// Package readline - Input-Verarbeitungsmodul
//
// Dieses Modul enthält die Hauptlogik für die Verarbeitung von
// Tastatureingaben in der Readline-Schleife.
//
// Hauptkomponenten:
// - processInput: Verarbeitet Escape-Sequenzen und Extended-Keys
// - processCharacter: Verarbeitet normale Zeichen und Steuersequenzen

package readline

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// processEscapeEx verarbeitet erweiterte Escape-Sequenzen (Pfeiltasten, etc.)
// Gibt true zurück wenn weiter iteriert werden soll
func (i *Instance) processEscapeEx(r rune, buf *Buffer, currentLineBuf *[]rune, escex *bool, metaDel *bool) (bool, error) {
	*escex = false

	switch r {
	case KeyUp:
		i.historyPrev(buf, currentLineBuf)
	case KeyDown:
		i.historyNext(buf, currentLineBuf)
	case KeyLeft:
		buf.MoveLeft()
	case KeyRight:
		buf.MoveRight()
	case CharBracketedPaste:
		var code string
		for range 3 {
			r, err := i.Terminal.Read()
			if err != nil {
				return false, io.EOF
			}
			code += string(r)
		}
		if code == CharBracketedPasteStart {
			i.Pasting = true
		} else if code == CharBracketedPasteEnd {
			i.Pasting = false
		}
	case KeyDel:
		if buf.DisplaySize() > 0 {
			buf.Delete()
		}
		*metaDel = true
	case MetaStart:
		buf.MoveToStart()
	case MetaEnd:
		buf.MoveToEnd()
	default:
		// Unbekannte Tasten überspringen
		return true, nil
	}
	return true, nil
}

// processEscape verarbeitet einfache Escape-Sequenzen (Alt+Buchstabe)
// Gibt true zurück wenn weiter iteriert werden soll
func (i *Instance) processEscape(r rune, buf *Buffer, esc *bool, escex *bool) bool {
	*esc = false

	switch r {
	case 'b':
		buf.MoveLeftWord()
	case 'f':
		buf.MoveRightWord()
	case CharBackspace:
		buf.DeleteWord()
	case CharEscapeEx:
		*escex = true
	}
	return true
}

// processCharacter verarbeitet normale Zeichen und Steuersequenzen
// Gibt (output, fertig, error) zurück
func (i *Instance) processCharacter(r rune, buf *Buffer, currentLineBuf *[]rune, esc *bool, metaDel *bool, draining bool) (string, bool, error) {
	switch r {
	case CharNull:
		return "", false, nil
	case CharEsc:
		*esc = true
	case CharInterrupt:
		i.pastedLines = nil
		i.Prompt.UseAlt = false
		return "", true, ErrInterrupt
	case CharPrev:
		i.historyPrev(buf, currentLineBuf)
	case CharNext:
		i.historyNext(buf, currentLineBuf)
	case CharLineStart:
		buf.MoveToStart()
	case CharLineEnd:
		buf.MoveToEnd()
	case CharBackward:
		buf.MoveLeft()
	case CharForward:
		buf.MoveRight()
	case CharBackspace, CharCtrlH:
		i.handleBackspace(buf)
	case CharTab:
		// Tab als 8 Leerzeichen
		for range 8 {
			buf.Add(' ')
		}
	case CharDelete:
		if buf.DisplaySize() > 0 {
			buf.Delete()
		} else {
			return "", true, io.EOF
		}
	case CharKill:
		buf.DeleteRemaining()
	case CharCtrlU:
		buf.DeleteBefore()
	case CharCtrlL:
		buf.ClearScreen()
	case CharCtrlW:
		buf.DeleteWord()
	case CharCtrlZ:
		fd := os.Stdin.Fd()
		output, err := handleCharCtrlZ(fd, i.Terminal.termios)
		return output, true, err
	case CharCtrlJ:
		// Wenn nicht im Cooked-Mode-Drain, als Multiline behandeln
		if !draining {
			i.pastedLines = append(i.pastedLines, buf.String())
			buf.Buf.Clear()
			buf.Pos = 0
			buf.DisplayPos = 0
			buf.LineHasSpace.Clear()
			fmt.Println()
			fmt.Print(i.Prompt.AltPrompt)
			i.Prompt.UseAlt = true
			return "", false, nil
		}
		// Cooked-Mode-Input: \n als Submit behandeln
		fallthrough
	case CharEnter:
		return i.handleEnter(buf), true, nil
	default:
		if *metaDel {
			*metaDel = false
			return "", false, nil
		}
		if r >= CharSpace || r == CharEnter || r == CharCtrlJ {
			buf.Add(r)
		}
	}
	return "", false, nil
}

// handleBackspace behandelt Backspace-Eingaben inkl. Multiline-Unterstützung
func (i *Instance) handleBackspace(buf *Buffer) {
	if buf.IsEmpty() && len(i.pastedLines) > 0 {
		lastIdx := len(i.pastedLines) - 1
		prevLine := i.pastedLines[lastIdx]
		i.pastedLines = i.pastedLines[:lastIdx]
		fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + ClearToEOL)
		if len(i.pastedLines) == 0 {
			fmt.Print(i.Prompt.Prompt)
			i.Prompt.UseAlt = false
		} else {
			fmt.Print(i.Prompt.AltPrompt)
		}
		for _, r := range prevLine {
			buf.Add(r)
		}
	} else {
		buf.Remove()
	}
}

// handleEnter verarbeitet die Enter-Taste und gibt den finalen Output zurück
func (i *Instance) handleEnter(buf *Buffer) string {
	output := buf.String()
	if len(i.pastedLines) > 0 {
		output = strings.Join(i.pastedLines, "\n") + "\n" + output
		i.pastedLines = nil
	}
	if output != "" {
		i.History.Add(output)
	}
	buf.MoveToEnd()
	fmt.Println()
	i.Prompt.UseAlt = false
	return output
}
