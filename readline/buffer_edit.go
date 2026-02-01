// Buffer-Edit-Modul: Bearbeitungsfunktionen fuer den Textpuffer
// Dieses Modul enthaelt alle Funktionen zum Hinzufuegen und Entfernen von Text.
// Aufgeteilt aus der urspruenglichen buffer.go (527 LOC)

package readline

import (
	"fmt"

	"github.com/mattn/go-runewidth"
)

func (b *Buffer) Add(r rune) {
	if b.Pos == b.Buf.Size() {
		b.AddChar(r, false)
	} else {
		b.AddChar(r, true)
	}
}

func (b *Buffer) AddChar(r rune, insert bool) {
	rLength := runewidth.RuneWidth(r)
	b.DisplayPos += rLength

	if b.Pos > 0 {
		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Printf("%c", r)
			fmt.Printf("\n%s", b.Prompt.AltPrompt)

			if insert {
				b.LineHasSpace.Set(b.DisplayPos/b.LineWidth-1, false)
			} else {
				b.LineHasSpace.Add(false)
			}

			// this case occurs when a double-width rune crosses the line boundary
		} else if b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth {
			if insert {
				fmt.Print(ClearToEOL)
			}
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			b.DisplayPos += 1
			fmt.Printf("%c", r)

			if insert {
				b.LineHasSpace.Set(b.DisplayPos/b.LineWidth-1, true)
			} else {
				b.LineHasSpace.Add(true)
			}
		} else {
			fmt.Printf("%c", r)
		}
	} else {
		fmt.Printf("%c", r)
	}

	if insert {
		b.Buf.Insert(b.Pos, r)
	} else {
		b.Buf.Add(r)
	}

	b.Pos += 1

	if insert {
		b.drawRemaining()
	}
}

func (b *Buffer) Remove() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		if r, ok := b.Buf.Get(b.Pos - 1); ok {
			rLength := runewidth.RuneWidth(r)
			hasSpace := b.GetLineSpacing(b.DisplayPos/b.LineWidth - 1)

			if b.DisplayPos%b.LineWidth == 0 {
				// if the user backspaces over the word boundary, do this magic to clear the line
				// and move to the end of the previous line
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.DisplaySize()%b.LineWidth < (b.DisplaySize()-rLength)%b.LineWidth {
					b.LineHasSpace.Remove(b.DisplayPos/b.LineWidth - 1)
				}

				if hasSpace {
					b.DisplayPos -= 1
					fmt.Print(CursorLeft)
				}

				if rLength == 2 {
					fmt.Print(CursorLeft + "  " + CursorLeftN(2))
				} else {
					fmt.Print(" " + CursorLeft)
				}
			} else if (b.DisplayPos-rLength)%b.LineWidth == 0 && hasSpace {
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.Pos == b.Buf.Size() {
					b.LineHasSpace.Remove(b.DisplayPos/b.LineWidth - 1)
				}
				b.DisplayPos -= 1
			} else {
				fmt.Print(CursorLeftN(rLength))
				for range rLength {
					fmt.Print(" ")
				}
				fmt.Print(CursorLeftN(rLength))
			}

			var eraseExtraLine bool
			if (b.DisplaySize()-1)%b.LineWidth == 0 || (rLength == 2 && ((b.DisplaySize()-2)%b.LineWidth == 0)) || b.DisplaySize()%b.LineWidth == 0 {
				eraseExtraLine = true
			}

			b.Pos -= 1
			b.DisplayPos -= rLength
			b.Buf.Remove(b.Pos)

			if b.Pos < b.Buf.Size() {
				b.drawRemaining()
				// this erases a line which is left over when backspacing in the middle of a line and there
				// are trailing characters which go over the line width boundary
				if eraseExtraLine {
					remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
					fmt.Print(CursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
					place := b.DisplayPos % b.LineWidth
					fmt.Print(CursorUpN(remainingLines+1) + CursorRightN(place+len(b.Prompt.prompt())))
				}
			}
		}
	}
}

func (b *Buffer) Delete() {
	if b.Buf.Size() > 0 && b.Pos < b.Buf.Size() {
		b.Buf.Remove(b.Pos)
		b.drawRemaining()
		if b.DisplaySize()%b.LineWidth == 0 {
			if b.DisplayPos != b.DisplaySize() {
				remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
				fmt.Print(CursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.DisplayPos % b.LineWidth
				fmt.Print(CursorUpN(remainingLines) + CursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) DeleteBefore() {
	if b.Pos > 0 {
		for cnt := b.Pos - 1; cnt >= 0; cnt-- {
			b.Remove()
		}
	}
}

func (b *Buffer) DeleteRemaining() {
	if b.DisplaySize() > 0 && b.Pos < b.DisplaySize() {
		charsToDel := b.Buf.Size() - b.Pos
		for range charsToDel {
			b.Delete()
		}
	}
}

func (b *Buffer) DeleteWord() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if !foundNonspace {
					b.Remove()
				} else {
					break
				}
			} else {
				foundNonspace = true
				b.Remove()
			}

			if b.Pos == 0 {
				break
			}
		}
	}
}
