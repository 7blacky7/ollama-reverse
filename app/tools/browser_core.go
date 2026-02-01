//go:build windows || darwin

// browser_core.go - Basis-Typen, State-Management und Kern-Funktionen
// Hauptkomponenten:
// - BrowserState: Session-Verwaltung pro Chat
// - Browser: Kern-Struktur mit State-Zugriff
// - NewBrowser: Factory-Funktion

package tools

import (
	"fmt"
	"strings"
	"sync"

	"github.com/ollama/ollama/app/ui/responses"
)

// PageType definiert den Typ einer Browser-Seite
type PageType string

const (
	PageTypeSearchResults PageType = "initial_results"
	PageTypeWebpage       PageType = "webpage"
)

// DefaultViewTokens ist die Anzahl der Tokens für die Model-Anzeige
const DefaultViewTokens = 1024

/*
Der Browser-Tool bietet Web-Browsing-Fähigkeiten für gpt-oss.
Das Model nutzt das Tool üblicherweise für Suche, dann Seite öffnen,
Text finden oder erneut suchen.

Jede Aktion wird im append-only PageStack gespeichert um die
Browser-Session-History zu verfolgen.

Execute() gibt den vollständigen Browser-State zurück. ui.go verwaltet
die State-Repräsentation zwischen Tool, UI und DB.

Ein neues Browser-Objekt wird pro Request erstellt - ui.go rekonstruiert
den State und übergibt die gestickte History.
*/

// BrowserState verwaltet die Browser-Session pro Chat
type BrowserState struct {
	mu   sync.RWMutex
	Data *responses.BrowserStateData
}

// Browser ist die Kern-Struktur für Browser-Operationen
type Browser struct {
	state *BrowserState
}

// State gibt den aktuellen Browser-State zurück (thread-safe)
func (b *Browser) State() *responses.BrowserStateData {
	b.state.mu.RLock()
	defer b.state.mu.RUnlock()
	return b.state.Data
}

// savePage speichert eine Seite im State
func (b *Browser) savePage(page *responses.Page) {
	b.state.Data.URLToPage[page.URL] = page
	b.state.Data.PageStack = append(b.state.Data.PageStack, page.URL)
}

// getPageFromStack holt eine Seite anhand der URL aus dem Cache
func (b *Browser) getPageFromStack(url string) (*responses.Page, error) {
	page, ok := b.state.Data.URLToPage[url]
	if !ok {
		return nil, fmt.Errorf("page not found for url %s", url)
	}
	return page, nil
}

// NewBrowser erstellt eine neue Browser-Instanz
func NewBrowser(state *responses.BrowserStateData) *Browser {
	if state == nil {
		state = &responses.BrowserStateData{
			PageStack:  []string{},
			ViewTokens: DefaultViewTokens,
			URLToPage:  make(map[string]*responses.Page),
		}
	}
	b := &BrowserState{
		Data: state,
	}

	return &Browser{
		state: b,
	}
}

// getEndLoc berechnet die End-Position für den Viewport basierend auf Token-Limits
func (b *Browser) getEndLoc(loc, numLines, totalLines int, lines []string) int {
	if numLines <= 0 {
		txt := b.joinLinesWithNumbers(lines[loc:])

		if len(txt) > b.state.Data.ViewTokens {
			maxCharsPerToken := 128
			upperBound := min((b.state.Data.ViewTokens+1)*maxCharsPerToken, len(txt))
			textToAnalyze := txt[:upperBound]
			approxTokens := len(textToAnalyze) / 4

			if approxTokens > b.state.Data.ViewTokens {
				endIdx := min(b.state.Data.ViewTokens*4, len(txt))
				numLines = strings.Count(txt[:endIdx], "\n") + 1
			} else {
				numLines = totalLines
			}
		} else {
			numLines = totalLines
		}
	}

	return min(loc+numLines, totalLines)
}

// joinLinesWithNumbers erstellt einen String mit Zeilennummern
func (b *Browser) joinLinesWithNumbers(lines []string) string {
	var builder strings.Builder
	var hadZeroLine bool
	for i, line := range lines {
		if i == 0 {
			builder.WriteString("L0:\n")
			hadZeroLine = true
		}
		if hadZeroLine {
			builder.WriteString(fmt.Sprintf("L%d: %s\n", i+1, line))
		} else {
			builder.WriteString(fmt.Sprintf("L%d: %s\n", i, line))
		}
	}
	return builder.String()
}

// displayPage formatiert und gibt die Seitenanzeige für das Model zurück
func (b *Browser) displayPage(page *responses.Page, cursor, loc, numLines int) (string, error) {
	totalLines := len(page.Lines)

	if loc >= totalLines {
		return "", fmt.Errorf("invalid location: %d (max: %d)", loc, totalLines-1)
	}

	endLoc := b.getEndLoc(loc, numLines, totalLines, page.Lines)

	var displayBuilder strings.Builder
	displayBuilder.WriteString(fmt.Sprintf("[%d] %s", cursor, page.Title))
	if page.URL != "" {
		displayBuilder.WriteString(fmt.Sprintf("(%s)\n", page.URL))
	} else {
		displayBuilder.WriteString("\n")
	}
	displayBuilder.WriteString(fmt.Sprintf("**viewing lines [%d - %d] of %d**\n\n", loc, endLoc-1, totalLines-1))

	var hadZeroLine bool
	for i := loc; i < endLoc; i++ {
		if i == 0 {
			displayBuilder.WriteString("L0:\n")
			hadZeroLine = true
		}
		if hadZeroLine {
			displayBuilder.WriteString(fmt.Sprintf("L%d: %s\n", i+1, page.Lines[i]))
		} else {
			displayBuilder.WriteString(fmt.Sprintf("L%d: %s\n", i, page.Lines[i]))
		}
	}

	return displayBuilder.String(), nil
}

