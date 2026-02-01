// Modul: selector_prompt.go
// Beschreibung: Oeffentliche API-Funktionen für Benutzer-Prompts.
// Enthält selectPrompt, multiSelectPrompt und confirmPrompt.

package config

import (
	"fmt"
	"os"

	"golang.org/x/term"
)

// selectPrompt prompts the user to select a single item from a list.
func selectPrompt(prompt string, items []selectItem) (string, error) {
	if len(items) == 0 {
		return "", fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return "", err
	}
	defer ts.restore()

	state := newSelectState(items)
	var lastLineCount int

	render := func() {
		clearLines(lastLineCount)
		lastLineCount = renderSelect(os.Stderr, prompt, state)
	}

	render()

	for {
		event, char, err := parseInput(os.Stdin)
		if err != nil {
			return "", err
		}

		done, result, err := state.handleInput(event, char)
		if done {
			clearLines(lastLineCount)
			if err != nil {
				return "", err
			}
			return result, nil
		}

		render()
	}
}

// multiSelectPrompt prompts the user to select multiple items from a list.
func multiSelectPrompt(prompt string, items []selectItem, preChecked []string) ([]string, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return nil, err
	}
	defer ts.restore()

	state := newMultiSelectState(items, preChecked)
	var lastLineCount int

	render := func() {
		clearLines(lastLineCount)
		lastLineCount = renderMultiSelect(os.Stderr, prompt, state)
	}

	render()

	for {
		event, char, err := parseInput(os.Stdin)
		if err != nil {
			return nil, err
		}

		done, result, err := state.handleInput(event, char)
		if done {
			clearLines(lastLineCount)
			if err != nil {
				return nil, err
			}
			return result, nil
		}

		render()
	}
}

func confirmPrompt(prompt string) (bool, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprintf(os.Stderr, "%s (\033[1my\033[0m/n) ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13:
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3:
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}
