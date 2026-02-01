//go:build windows || darwin

// browser_actions.go - Browser-Aktionen: Seiten öffnen und Text suchen
// Hauptkomponenten:
// - BrowserOpen: Tool zum Öffnen von Links/URLs
// - BrowserFind: Tool zum Suchen von Text auf der Seite

package tools

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/app/ui/responses"
)

// BrowserOpen ist das Tool zum Öffnen von Links
type BrowserOpen struct {
	Browser
	crawlPage *BrowserCrawler
}

// NewBrowserOpen erstellt eine neue BrowserOpen-Instanz
func NewBrowserOpen(bb *Browser) *BrowserOpen {
	if bb == nil {
		bb = &Browser{
			state: &BrowserState{
				Data: &responses.BrowserStateData{
					PageStack:  []string{},
					ViewTokens: DefaultViewTokens,
					URLToPage:  make(map[string]*responses.Page),
				},
			},
		}
	}
	return &BrowserOpen{
		Browser:   *bb,
		crawlPage: &BrowserCrawler{},
	}
}

// Name gibt den Tool-Namen zurück
func (b *BrowserOpen) Name() string {
	return "browser.open"
}

// Description gibt die Tool-Beschreibung zurück
func (b *BrowserOpen) Description() string {
	return "Open a link in the browser"
}

// Prompt gibt den Tool-Prompt zurück
func (b *BrowserOpen) Prompt() string {
	return ""
}

// Schema gibt das Tool-Schema zurück
func (b *BrowserOpen) Schema() map[string]any {
	return map[string]any{}
}

// Execute führt das Öffnen einer Seite aus
func (b *BrowserOpen) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	cursor, loc, numLines := b.parseOpenArgs(args)

	// Seite vom Cursor holen
	page, err := b.getPageByCursor(cursor)
	if err != nil && cursor >= 0 {
		return nil, "", err
	}

	// Versuchen id als String (URL) zu holen
	if url, ok := args["id"].(string); ok {
		return b.openByURL(ctx, url, loc, numLines)
	}

	// Versuchen id als Integer (Link-ID von aktueller Seite) zu holen
	if id, ok := args["id"].(float64); ok {
		return b.openByLinkID(ctx, page, int(id), loc, numLines)
	}

	// Wenn keine id angegeben, aktuelle Seite anzeigen
	if page == nil {
		return nil, "", fmt.Errorf("no current page to display")
	}
	b.state.Data.PageStack = append(b.state.Data.PageStack, page.URL)
	cursor = len(b.state.Data.PageStack) - 1

	pageText, err := b.displayPage(page, cursor, loc, numLines)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}
	return b.state.Data, pageText, nil
}

// parseOpenArgs extrahiert cursor, loc und numLines aus den Argumenten
func (b *BrowserOpen) parseOpenArgs(args map[string]any) (int, int, int) {
	cursor := -1
	if c, ok := args["cursor"].(float64); ok {
		cursor = int(c)
	} else if c, ok := args["cursor"].(int); ok {
		cursor = c
	}

	loc := 0
	if l, ok := args["loc"].(float64); ok {
		loc = int(l)
	} else if l, ok := args["loc"].(int); ok {
		loc = l
	}

	numLines := -1
	if n, ok := args["num_lines"].(float64); ok {
		numLines = int(n)
	} else if n, ok := args["num_lines"].(int); ok {
		numLines = n
	}

	return cursor, loc, numLines
}

// getPageByCursor holt eine Seite anhand des Cursors
func (b *BrowserOpen) getPageByCursor(cursor int) (*responses.Page, error) {
	if cursor >= 0 {
		if cursor >= len(b.state.Data.PageStack) {
			return nil, fmt.Errorf("cursor %d is out of range (pageStack length: %d)", cursor, len(b.state.Data.PageStack))
		}
		return b.getPageFromStack(b.state.Data.PageStack[cursor])
	}

	if len(b.state.Data.PageStack) != 0 {
		pageURL := b.state.Data.PageStack[len(b.state.Data.PageStack)-1]
		return b.getPageFromStack(pageURL)
	}
	return nil, nil
}

// openByURL öffnet eine Seite anhand der URL
func (b *BrowserOpen) openByURL(ctx context.Context, url string, loc, numLines int) (any, string, error) {
	if existingPage, ok := b.state.Data.URLToPage[url]; ok {
		b.savePage(existingPage)
		cursor := len(b.state.Data.PageStack) - 1
		pageText, err := b.displayPage(existingPage, cursor, loc, numLines)
		if err != nil {
			return nil, "", fmt.Errorf("failed to display page: %w", err)
		}
		return b.state.Data, pageText, nil
	}

	if b.crawlPage == nil {
		b.crawlPage = &BrowserCrawler{}
	}
	crawlResponse, err := b.crawlPage.Execute(ctx, map[string]any{
		"urls":   []any{url},
		"latest": false,
	})
	if err != nil {
		return nil, "", fmt.Errorf("failed to crawl URL %s: %w", url, err)
	}

	newPage, err := b.buildPageFromCrawlResult(url, crawlResponse)
	if err != nil {
		return nil, "", fmt.Errorf("failed to build page from crawl result: %w", err)
	}

	b.savePage(newPage)
	cursor := len(b.state.Data.PageStack) - 1
	pageText, err := b.displayPage(newPage, cursor, loc, numLines)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}
	return b.state.Data, pageText, nil
}

// openByLinkID öffnet eine Seite anhand der Link-ID auf der aktuellen Seite
func (b *BrowserOpen) openByLinkID(ctx context.Context, page *responses.Page, idInt, loc, numLines int) (any, string, error) {
	if page == nil {
		return nil, "", fmt.Errorf("no current page to resolve link from")
	}

	pageURL, ok := page.Links[idInt]
	if !ok {
		return nil, "", fmt.Errorf("invalid link id %d", idInt)
	}

	newPage, ok := b.state.Data.URLToPage[pageURL]
	if !ok {
		if b.crawlPage == nil {
			b.crawlPage = &BrowserCrawler{}
		}
		crawlResponse, err := b.crawlPage.Execute(ctx, map[string]any{
			"urls":   []any{pageURL},
			"latest": false,
		})
		if err != nil {
			return nil, "", fmt.Errorf("failed to crawl URL %s: %w", pageURL, err)
		}

		newPage, err = b.buildPageFromCrawlResult(pageURL, crawlResponse)
		if err != nil {
			return nil, "", fmt.Errorf("failed to build page from crawl result: %w", err)
		}
	}

	b.savePage(newPage)
	cursor := len(b.state.Data.PageStack) - 1
	pageText, err := b.displayPage(newPage, cursor, loc, numLines)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}
	return b.state.Data, pageText, nil
}

// BrowserFind ist das Tool zum Suchen von Text auf Seiten
type BrowserFind struct {
	Browser
}

// NewBrowserFind erstellt eine neue BrowserFind-Instanz
func NewBrowserFind(bb *Browser) *BrowserFind {
	return &BrowserFind{
		Browser: *bb,
	}
}

// Name gibt den Tool-Namen zurück
func (b *BrowserFind) Name() string {
	return "browser.find"
}

// Description gibt die Tool-Beschreibung zurück
func (b *BrowserFind) Description() string {
	return "Find a term in the browser"
}

// Prompt gibt den Tool-Prompt zurück
func (b *BrowserFind) Prompt() string {
	return ""
}

// Schema gibt das Tool-Schema zurück
func (b *BrowserFind) Schema() map[string]any {
	return map[string]any{}
}

// Execute führt die Text-Suche auf einer Seite aus
func (b *BrowserFind) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	pattern, ok := args["pattern"].(string)
	if !ok {
		return nil, "", fmt.Errorf("pattern parameter is required")
	}

	cursor := -1
	if c, ok := args["cursor"].(float64); ok {
		cursor = int(c)
	}

	page, err := b.getPageForFind(cursor)
	if err != nil {
		return nil, "", err
	}

	findPage := buildFindResultsPage(pattern, page)
	b.savePage(findPage)
	newCursor := len(b.state.Data.PageStack) - 1

	pageText, err := b.displayPage(findPage, newCursor, 0, -1)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}

	return b.state.Data, pageText, nil
}

// getPageForFind holt die Seite für die Suche
func (b *BrowserFind) getPageForFind(cursor int) (*responses.Page, error) {
	if cursor == -1 {
		if len(b.state.Data.PageStack) == 0 {
			return nil, fmt.Errorf("no pages to search in")
		}
		page, err := b.getPageFromStack(b.state.Data.PageStack[len(b.state.Data.PageStack)-1])
		if err != nil {
			return nil, fmt.Errorf("page not found for cursor %d: %w", cursor, err)
		}
		return page, nil
	}

	if cursor < 0 || cursor >= len(b.state.Data.PageStack) {
		return nil, fmt.Errorf("cursor %d is out of range [0-%d]", cursor, len(b.state.Data.PageStack)-1)
	}
	page, err := b.getPageFromStack(b.state.Data.PageStack[cursor])
	if err != nil {
		return nil, fmt.Errorf("page not found for cursor %d: %w", cursor, err)
	}
	return page, nil
}
