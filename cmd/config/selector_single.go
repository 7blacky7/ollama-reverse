// Modul: selector_single.go
// Beschreibung: Single-Select State-Management und Input-Handling.
// Verwaltet den Zustand für Einzelauswahl-Listen mit Filter und Scrolling.

package config

type selectState struct {
	items        []selectItem
	filter       string
	selected     int
	scrollOffset int
}

func newSelectState(items []selectItem) *selectState {
	return &selectState{items: items}
}

func (s *selectState) filtered() []selectItem {
	return filterItems(s.items, s.filter)
}

func (s *selectState) handleInput(event inputEvent, char byte) (done bool, result string, err error) {
	filtered := s.filtered()

	switch event {
	case eventEnter:
		if len(filtered) > 0 && s.selected < len(filtered) {
			return true, filtered[s.selected].Name, nil
		}
	case eventEscape:
		return true, "", errCancelled
	case eventBackspace:
		if len(s.filter) > 0 {
			s.filter = s.filter[:len(s.filter)-1]
			s.selected = 0
			s.scrollOffset = 0
		}
	case eventUp:
		if s.selected > 0 {
			s.selected--
			if s.selected < s.scrollOffset {
				s.scrollOffset = s.selected
			}
		}
	case eventDown:
		if s.selected < len(filtered)-1 {
			s.selected++
			if s.selected >= s.scrollOffset+maxDisplayedItems {
				s.scrollOffset = s.selected - maxDisplayedItems + 1
			}
		}
	case eventChar:
		s.filter += string(char)
		s.selected = 0
		s.scrollOffset = 0
	}

	return false, "", nil
}
