// Buffer-Modul: Hauptstruktur und Basis-Funktionen
// Dieses Modul verwaltet den Textpuffer fuer die Readline-Eingabe.
// Aufgeteilt aus der urspruenglichen buffer.go (527 LOC)
// Siehe auch: buffer_cursor.go, buffer_edit.go

package readline

import (
	"fmt"
	"os"

	"github.com/emirpasic/gods/v2/lists/arraylist"
	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

type Buffer struct {
	DisplayPos int
	Pos        int
	Buf        *arraylist.List[rune]
	// LineHasSpace is an arraylist of bools to keep track of whether a line has a space at the end
	LineHasSpace *arraylist.List[bool]
	Prompt       *Prompt
	LineWidth    int
	Width        int
	Height       int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height := 80, 24
	if termWidth, termHeight, err := term.GetSize(fd); err == nil {
		width, height = termWidth, termHeight
	}

	lwidth := width - len(prompt.prompt())

	b := &Buffer{
		DisplayPos:   0,
		Pos:          0,
		Buf:          arraylist.New[rune](),
		LineHasSpace: arraylist.New[bool](),
		Prompt:       prompt,
		Width:        width,
		Height:       height,
		LineWidth:    lwidth,
	}

	return b, nil
}

func (b *Buffer) GetLineSpacing(line int) bool {
	hasSpace, _ := b.LineHasSpace.Get(line)
	return hasSpace
}

func (b *Buffer) DisplaySize() int {
	sum := 0
	for i := range b.Buf.Size() {
		if r, ok := b.Buf.Get(i); ok {
			sum += runewidth.RuneWidth(r)
		}
	}

	return sum
}

func (b *Buffer) IsEmpty() bool {
	return b.Buf.Empty()
}

func (b *Buffer) String() string {
	return b.StringN(0)
}

func (b *Buffer) StringN(n int) string {
	return b.StringNM(n, 0)
}

func (b *Buffer) StringNM(n, m int) string {
	var s string
	if m == 0 {
		m = b.Buf.Size()
	}
	for cnt := n; cnt < m; cnt++ {
		c, _ := b.Buf.Get(cnt)
		s += string(c)
	}
	return s
}

func (b *Buffer) ClearScreen() {
	fmt.Print(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.DisplayPos
		currIndex := b.Pos
		b.Pos = 0
		b.DisplayPos = 0
		b.drawRemaining()
		fmt.Print(CursorReset + CursorRightN(len(b.Prompt.prompt())))
		if currPos > 0 {
			targetLine := currPos / b.LineWidth
			if targetLine > 0 {
				for range targetLine {
					fmt.Print(CursorDown)
				}
			}
			remainder := currPos % b.LineWidth
			if remainder > 0 {
				fmt.Print(CursorRightN(remainder))
			}
			if currPos%b.LineWidth == 0 {
				fmt.Print(CursorBOL + b.Prompt.AltPrompt)
			}
		}
		b.Pos = currIndex
		b.DisplayPos = currPos
	}
}

func (b *Buffer) Replace(r []rune) {
	b.DisplayPos = 0
	b.Pos = 0
	lineNums := b.DisplaySize() / b.LineWidth

	b.Buf.Clear()

	fmt.Print(CursorBOL + ClearToEOL)

	for range lineNums {
		fmt.Print(CursorUp + CursorBOL + ClearToEOL)
	}

	fmt.Print(CursorBOL + b.Prompt.prompt())

	for _, c := range r {
		b.Add(c)
	}
}

func (b *Buffer) countRemainingLineWidth(place int) int {
	var sum int
	counter := -1
	var prevLen int

	for place <= b.LineWidth {
		counter += 1
		sum += prevLen
		if r, ok := b.Buf.Get(b.Pos + counter); ok {
			place += runewidth.RuneWidth(r)
			prevLen = len(string(r))
		} else {
			break
		}
	}

	return sum
}

func (b *Buffer) drawRemaining() {
	var place int
	remainingText := b.StringN(b.Pos)
	if b.Pos > 0 {
		place = b.DisplayPos % b.LineWidth
	}
	fmt.Print(CursorHide)

	// render the rest of the current line
	currLineLength := b.countRemainingLineWidth(place)

	currLine := remainingText[:min(currLineLength, len(remainingText))]
	currLineSpace := runewidth.StringWidth(currLine)
	remLength := runewidth.StringWidth(remainingText)

	if len(currLine) > 0 {
		fmt.Print(ClearToEOL + currLine + CursorLeftN(currLineSpace))
	} else {
		fmt.Print(ClearToEOL)
	}

	if currLineSpace != b.LineWidth-place && currLineSpace != remLength {
		b.LineHasSpace.Set(b.DisplayPos/b.LineWidth, true)
	} else if currLineSpace != b.LineWidth-place {
		b.LineHasSpace.Remove(b.DisplayPos / b.LineWidth)
	} else {
		b.LineHasSpace.Set(b.DisplayPos/b.LineWidth, false)
	}

	if (b.DisplayPos+currLineSpace)%b.LineWidth == 0 && currLine == remainingText {
		fmt.Print(CursorRightN(currLineSpace))
		fmt.Printf("\n%s", b.Prompt.AltPrompt)
		fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width-currLineSpace))
	}

	// render the other lines
	if remLength > currLineSpace {
		remaining := (remainingText[len(currLine):])
		var totalLines int
		var displayLength int
		var lineLength int = currLineSpace

		for _, c := range remaining {
			if displayLength == 0 || (displayLength+runewidth.RuneWidth(c))%b.LineWidth < displayLength%b.LineWidth {
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				totalLines += 1

				if displayLength != 0 {
					if lineLength == b.LineWidth {
						b.LineHasSpace.Set(b.DisplayPos/b.LineWidth+totalLines-1, false)
					} else {
						b.LineHasSpace.Set(b.DisplayPos/b.LineWidth+totalLines-1, true)
					}
				}

				lineLength = 0
			}

			displayLength += runewidth.RuneWidth(c)
			lineLength += runewidth.RuneWidth(c)
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL + CursorUpN(totalLines) + CursorBOL + CursorRightN(b.Width-currLineSpace))

		hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)

		if hasSpace && b.DisplayPos%b.LineWidth != b.LineWidth-1 {
			fmt.Print(CursorLeft)
		}
	}

	fmt.Print(CursorShow)
}
