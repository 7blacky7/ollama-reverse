//go:build windows || darwin

// browser_utils.go - Hilfsfunktionen für Browser-Tools
// Hauptkomponenten:
// - processMarkdownLinks: Konvertiert Markdown-Links ins 【id†text†domain】 Format
// - wrapLines: Bricht Text in Zeilen mit maximaler Breite um
// - buildFindResultsPage: Erstellt Suchergebnis-Seiten für Text-Suche

package tools

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/ollama/ollama/app/ui/responses"
)

// processMarkdownLinks konvertiert Markdown-Links in nummerierte Referenzen
// Gibt den verarbeiteten Text und eine Map der Link-IDs zu URLs zurück
func processMarkdownLinks(text string) (string, map[int]string) {
	links := make(map[int]string)

	// Immer bei 0 starten für konsistente Nummerierung
	linkID := 0

	// Erst mehrzeilige Markdown-Links zusammenfügen
	multiLinePattern := regexp.MustCompile(`\[([^\]]+)\]\s*\n\s*\(([^)]+)\)`)
	text = multiLinePattern.ReplaceAllStringFunc(text, func(match string) string {
		// Zeilenumbrüche durch Leerzeichen ersetzen
		cleaned := strings.ReplaceAll(match, "\n", " ")
		// Mehrfache Leerzeichen entfernen
		cleaned = regexp.MustCompile(`\s+`).ReplaceAllString(cleaned, " ")
		return cleaned
	})

	// Alle Markdown-Links verarbeiten (inkl. der bereinigten mehrzeiligen)
	linkPattern := regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)

	processedText := linkPattern.ReplaceAllStringFunc(text, func(match string) string {
		matches := linkPattern.FindStringSubmatch(match)
		if len(matches) != 3 {
			return match
		}

		linkText := strings.TrimSpace(matches[1])
		linkURL := strings.TrimSpace(matches[2])

		// Domain aus URL extrahieren
		domain := linkURL
		if u, err := url.Parse(linkURL); err == nil && u.Host != "" {
			domain = u.Host
			// www. Präfix entfernen
			domain = strings.TrimPrefix(domain, "www.")
		}

		// Formatierter Link erstellen
		formatted := fmt.Sprintf("【%d†%s†%s】", linkID, linkText, domain)

		// Link speichern
		links[linkID] = linkURL
		linkID++

		return formatted
	})

	return processedText, links
}

// wrapLines bricht Text in Zeilen mit maximaler Breite um
func wrapLines(text string, width int) []string {
	if width <= 0 {
		width = 80
	}

	lines := strings.Split(text, "\n")
	var wrapped []string

	for _, line := range lines {
		if line == "" {
			wrapped = append(wrapped, "")
		} else if len(line) <= width {
			wrapped = append(wrapped, line)
		} else {
			words := strings.Fields(line)
			if len(words) == 0 {
				wrapped = append(wrapped, line)
				continue
			}

			currentLine := ""
			for _, word := range words {
				testLine := currentLine
				if testLine != "" {
					testLine += " "
				}
				testLine += word

				if len(testLine) > width && currentLine != "" {
					wrapped = append(wrapped, currentLine)
					currentLine = word
				} else {
					if currentLine != "" {
						currentLine += " "
					}
					currentLine += word
				}
			}

			if currentLine != "" {
				wrapped = append(wrapped, currentLine)
			}
		}
	}

	return wrapped
}

// buildFindResultsPage erstellt eine Seite mit den Text-Such-Ergebnissen
func buildFindResultsPage(pattern string, page *responses.Page) *responses.Page {
	findPage := &responses.Page{
		Title:     fmt.Sprintf("Find results for text: `%s` in `%s`", pattern, page.Title),
		URL:       fmt.Sprintf("find_results_%s", pattern),
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	var textBuilder strings.Builder
	matchIdx := 0
	maxResults := 50
	numShowLines := 4
	patternLower := strings.ToLower(pattern)

	var resultChunks []string
	lineIdx := 0

	for lineIdx < len(page.Lines) {
		line := page.Lines[lineIdx]
		lineLower := strings.ToLower(line)

		if !strings.Contains(lineLower, patternLower) {
			lineIdx++
			continue
		}

		endLine := min(lineIdx+numShowLines, len(page.Lines))

		var snippetBuilder strings.Builder
		for j := lineIdx; j < endLine; j++ {
			snippetBuilder.WriteString(page.Lines[j])
			if j < endLine-1 {
				snippetBuilder.WriteString("\n")
			}
		}
		snippet := snippetBuilder.String()

		linkFormat := fmt.Sprintf("【%d†match at L%d】", matchIdx, lineIdx)
		resultChunk := fmt.Sprintf("%s\n%s", linkFormat, snippet)
		resultChunks = append(resultChunks, resultChunk)

		if len(resultChunks) >= maxResults {
			break
		}

		matchIdx++
		lineIdx += numShowLines
	}

	if len(resultChunks) > 0 {
		textBuilder.WriteString(strings.Join(resultChunks, "\n\n"))
	}

	if matchIdx == 0 {
		findPage.Text = fmt.Sprintf("No `find` results for pattern: `%s`", pattern)
	} else {
		findPage.Text = textBuilder.String()
	}

	findPage.Lines = wrapLines(findPage.Text, 80)
	return findPage
}
