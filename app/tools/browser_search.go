//go:build windows || darwin

// browser_search.go - Web-Suche und Seiten-Aufbau Funktionalität
// Hauptkomponenten:
// - BrowserSearch: Tool für Web-Suche
// - processMarkdownLinks: Verarbeitet Markdown-Links
// - buildSearchResultsPage: Erstellt Suchergebnis-Seiten
// - buildPageFromCrawlResult: Erstellt Seiten aus Crawl-Ergebnissen

package tools

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/app/ui/responses"
)

// BrowserSearch ist das Tool für Web-Suche
type BrowserSearch struct {
	Browser
	webSearch *BrowserWebSearch
}

// NewBrowserSearch erstellt eine neue BrowserSearch-Instanz
func NewBrowserSearch(bb *Browser) *BrowserSearch {
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
	return &BrowserSearch{
		Browser:   *bb,
		webSearch: &BrowserWebSearch{},
	}
}

// Name gibt den Tool-Namen zurück
func (b *BrowserSearch) Name() string {
	return "browser.search"
}

// Description gibt die Tool-Beschreibung zurück
func (b *BrowserSearch) Description() string {
	return "Search the web for information"
}

// Prompt gibt den Tool-Prompt zurück
func (b *BrowserSearch) Prompt() string {
	return ""
}

// Schema gibt das Tool-Schema zurück
func (b *BrowserSearch) Schema() map[string]any {
	return map[string]any{}
}

// Execute führt die Web-Suche aus
func (b *BrowserSearch) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, "", fmt.Errorf("query parameter is required")
	}

	topn, ok := args["topn"].(int)
	if !ok {
		topn = 5
	}

	searchArgs := map[string]any{
		"queries":     []any{query},
		"max_results": topn,
	}

	result, err := b.webSearch.Execute(ctx, searchArgs)
	if err != nil {
		return nil, "", fmt.Errorf("search error: %w", err)
	}

	searchResponse, ok := result.(*WebSearchResponse)
	if !ok {
		return nil, "", fmt.Errorf("invalid search results format")
	}

	// Haupt-Suchergebnisseite erstellen
	searchResultsPage := b.buildSearchResultsPageCollection(query, searchResponse)
	b.savePage(searchResultsPage)
	cursor := len(b.state.Data.PageStack) - 1

	// Ergebnis für jede Seite cachen
	for _, queryResults := range searchResponse.Results {
		for i, result := range queryResults {
			resultPage := b.buildSearchResultsPage(&result, i+1)
			b.state.Data.URLToPage[resultPage.URL] = resultPage
		}
	}

	pageText, err := b.displayPage(searchResultsPage, cursor, 0, -1)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}

	return b.state.Data, pageText, nil
}

// buildSearchResultsPageCollection erstellt eine Übersichtsseite mit allen Suchergebnissen
func (b *Browser) buildSearchResultsPageCollection(query string, results *WebSearchResponse) *responses.Page {
	page := &responses.Page{
		URL:       "search_results_" + query,
		Title:     query,
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	var textBuilder strings.Builder
	linkIdx := 0

	textBuilder.WriteString("\n")
	textBuilder.WriteString("URL: \n")
	textBuilder.WriteString("# Search Results\n")
	textBuilder.WriteString("\n")

	for _, queryResults := range results.Results {
		for _, result := range queryResults {
			domain := result.URL
			if u, err := url.Parse(result.URL); err == nil && u.Host != "" {
				domain = u.Host
				domain = strings.TrimPrefix(domain, "www.")
			}

			linkFormat := fmt.Sprintf("* 【%d†%s†%s】", linkIdx, result.Title, domain)
			textBuilder.WriteString(linkFormat)

			numChars := min(len(result.Content.FullText), 400)
			snippet := strings.TrimSpace(result.Content.FullText[:numChars])
			textBuilder.WriteString(snippet)
			textBuilder.WriteString("\n")

			page.Links[linkIdx] = result.URL
			linkIdx++
		}
	}

	page.Text = textBuilder.String()
	page.Lines = wrapLines(page.Text, 80)

	return page
}

// buildSearchResultsPage erstellt eine einzelne Suchergebnis-Seite
func (b *Browser) buildSearchResultsPage(result *WebSearchResult, linkIdx int) *responses.Page {
	page := &responses.Page{
		URL:       result.URL,
		Title:     result.Title,
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	var textBuilder strings.Builder

	linkFormat := fmt.Sprintf("【%d†%s】", linkIdx, result.Title)
	textBuilder.WriteString(linkFormat)
	textBuilder.WriteString("\n")
	textBuilder.WriteString(fmt.Sprintf("URL: %s\n", result.URL))
	numChars := min(len(result.Content.FullText), 300)
	textBuilder.WriteString(result.Content.FullText[:numChars])
	textBuilder.WriteString("\n\n")

	if result.Content.FullText == "" {
		page.Links[linkIdx] = result.URL
	}

	if result.Content.FullText != "" {
		page.Text = fmt.Sprintf("URL: %s\n%s", result.URL, result.Content.FullText)
		processedText, processedLinks := processMarkdownLinks(page.Text)
		page.Text = processedText
		page.Links = processedLinks
	} else {
		page.Text = textBuilder.String()
	}

	page.Lines = wrapLines(page.Text, 80)

	return page
}

// buildPageFromCrawlResult erstellt eine Page aus Crawl-API-Ergebnissen
func (b *Browser) buildPageFromCrawlResult(requestedURL string, crawlResponse *CrawlResponse) (*responses.Page, error) {
	page := &responses.Page{
		URL:       requestedURL,
		Title:     requestedURL,
		Text:      "",
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	for url, urlResults := range crawlResponse.Results {
		if len(urlResults) > 0 {
			result := urlResults[0]

			if result.Content.FullText != "" {
				page.Text = result.Content.FullText
			}

			if result.Title != "" {
				page.Title = result.Title
			}

			page.URL = url

			for i, link := range result.Extras.Links {
				if link.Href != "" {
					page.Links[i] = link.Href
				} else if link.URL != "" {
					page.Links[i] = link.URL
				}
			}

			break
		}
	}

	if page.Text == "" {
		page.Text = "No content could be extracted from this page."
	} else {
		page.Text = fmt.Sprintf("URL: %s\n%s", page.URL, page.Text)
	}

	processedText, processedLinks := processMarkdownLinks(page.Text)
	page.Text = processedText
	page.Links = processedLinks
	page.Lines = wrapLines(page.Text, 80)

	return page, nil
}

