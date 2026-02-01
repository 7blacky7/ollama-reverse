// Modul: selector_multi.go
// Beschreibung: Multi-Select State-Management und Input-Handling.
// Verwaltet den Zustand für Mehrfachauswahl-Listen mit Checkboxen und Reihenfolge.

package config

type multiSelectState struct {
	items         []selectItem
	itemIndex     map[string]int
	filter        string
	highlighted   int
	scrollOffset  int
	checked       map[int]bool
	checkOrder    []int
	focusOnButton bool
}

func newMultiSelectState(items []selectItem, preChecked []string) *multiSelectState {
	s := &multiSelectState{
		items:     items,
		itemIndex: make(map[string]int, len(items)),
		checked:   make(map[int]bool),
	}

	for i, item := range items {
		s.itemIndex[item.Name] = i
	}

	for _, name := range preChecked {
		if idx, ok := s.itemIndex[name]; ok {
			s.checked[idx] = true
			s.checkOrder = append(s.checkOrder, idx)
		}
	}

	return s
}

func (s *multiSelectState) filtered() []selectItem {
	return filterItems(s.items, s.filter)
}

func (s *multiSelectState) toggleItem() {
	filtered := s.filtered()
	if len(filtered) == 0 || s.highlighted >= len(filtered) {
		return
	}

	item := filtered[s.highlighted]
	origIdx := s.itemIndex[item.Name]

	if s.checked[origIdx] {
		delete(s.checked, origIdx)
		for i, idx := range s.checkOrder {
			if idx == origIdx {
				s.checkOrder = append(s.checkOrder[:i], s.checkOrder[i+1:]...)
				break
			}
		}
	} else {
		s.checked[origIdx] = true
		s.checkOrder = append(s.checkOrder, origIdx)
	}
}

func (s *multiSelectState) handleInput(event inputEvent, char byte) (done bool, result []string, err error) {
	filtered := s.filtered()

	switch event {
	case eventEnter:
		if s.focusOnButton && len(s.checkOrder) > 0 {
			var res []string
			for _, idx := range s.checkOrder {
				res = append(res, s.items[idx].Name)
			}
			return true, res, nil
		} else if !s.focusOnButton {
			s.toggleItem()
		}
	case eventTab:
		if len(s.checkOrder) > 0 {
			s.focusOnButton = !s.focusOnButton
		}
	case eventEscape:
		return true, nil, errCancelled
	case eventBackspace:
		if len(s.filter) > 0 {
			s.filter = s.filter[:len(s.filter)-1]
			s.highlighted = 0
			s.scrollOffset = 0
			s.focusOnButton = false
		}
	case eventUp:
		if s.focusOnButton {
			s.focusOnButton = false
		} else if s.highlighted > 0 {
			s.highlighted--
			if s.highlighted < s.scrollOffset {
				s.scrollOffset = s.highlighted
			}
		}
	case eventDown:
		if s.focusOnButton {
			s.focusOnButton = false
		} else if s.highlighted < len(filtered)-1 {
			s.highlighted++
			if s.highlighted >= s.scrollOffset+maxDisplayedItems {
				s.scrollOffset = s.highlighted - maxDisplayedItems + 1
			}
		}
	case eventChar:
		s.filter += string(char)
		s.highlighted = 0
		s.scrollOffset = 0
		s.focusOnButton = false
	}

	return false, nil, nil
}

func (s *multiSelectState) selectedCount() int {
	return len(s.checkOrder)
}
