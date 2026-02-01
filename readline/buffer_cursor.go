// Buffer-Cursor-Modul: Cursor-Bewegungen im Textpuffer
// Dieses Modul enthaelt alle Funktionen zur Cursor-Navigation.
// Aufgeteilt aus der urspruenglichen buffer.go (527 LOC)

package readline

import (
	"fmt"

	"github.com/mattn/go-runewidth"
)

func (b *Buffer) MoveLeft() {
	if b.Pos > 0 {
		// asserts that we retrieve a rune
		if r, ok := b.Buf.Get(b.Pos - 1); ok {
			rLength := runewidth.RuneWidth(r)

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width))
				if rLength == 2 {
					fmt.Print(CursorLeft)
				}

				line := b.DisplayPos/b.LineWidth - 1
				hasSpace := b.GetLineSpacing(line)
				if hasSpace {
					b.DisplayPos -= 1
					fmt.Print(CursorLeft)
				}
			} else {
				fmt.Print(CursorLeftN(rLength))
			}

			b.Pos -= 1
			b.DisplayPos -= rLength
		}
	}
}

func (b *Buffer) MoveLeftWord() {
	if b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if foundNonspace {
					break
				}
			} else {
				foundNonspace = true
			}
			b.MoveLeft()

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) MoveRight() {
	if b.Pos < b.Buf.Size() {
		if r, ok := b.Buf.Get(b.Pos); ok {
			rLength := runewidth.RuneWidth(r)
			b.Pos += 1
			hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)
			b.DisplayPos += rLength

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
			} else if (b.DisplayPos-rLength)%b.LineWidth == b.LineWidth-1 && hasSpace {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())+rLength))
				b.DisplayPos += 1
			} else if b.LineHasSpace.Size() > 0 && b.DisplayPos%b.LineWidth == b.LineWidth-1 && hasSpace {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
				b.DisplayPos += 1
			} else {
				fmt.Print(CursorRightN(rLength))
			}
		}
	}
}

func (b *Buffer) MoveRightWord() {
	if b.Pos < b.Buf.Size() {
		for {
			b.MoveRight()
			v, _ := b.Buf.Get(b.Pos)
			if v == ' ' {
				break
			}

			if b.Pos == b.Buf.Size() {
				break
			}
		}
	}
}

func (b *Buffer) MoveToStart() {
	if b.Pos > 0 {
		currLine := b.DisplayPos / b.LineWidth
		if currLine > 0 {
			for range currLine {
				fmt.Print(CursorUp)
			}
		}
		fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		b.Pos = 0
		b.DisplayPos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.Pos < b.Buf.Size() {
		currLine := b.DisplayPos / b.LineWidth
		totalLines := b.DisplaySize() / b.LineWidth
		if currLine < totalLines {
			for range totalLines - currLine {
				fmt.Print(CursorDown)
			}
			remainder := b.DisplaySize() % b.LineWidth
			fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(CursorRightN(b.DisplaySize() - b.DisplayPos))
		}

		b.Pos = b.Buf.Size()
		b.DisplayPos = b.DisplaySize()
	}
}
