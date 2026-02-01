// Package parsers - JSON-Hilfsfunktionen für Ministral Parser
//
// Diese Datei enthält Hilfsfunktionen für das Parsen von JSON
// in Tool-Call-Argumenten.
//
// Hauptfunktionen:
// - findJSONEnd: Findet das Ende eines JSON-Objekts
//
// Hinweis: overlap und trailingWhitespaceLen sind in parsers.go definiert.
package parsers

// findJSONEnd findet den Index der schließenden Klammer eines JSON-Objekts.
// Behandelt korrekt verschachtelte Objekte, Arrays und Strings
// (einschließlich escaped Zeichen).
// Gibt -1 zurück wenn das JSON noch nicht vollständig ist.
func findJSONEnd(s string) int {
	depth := 0
	inString := false
	escaped := false

	for i, r := range s {
		if inString {
			switch {
			case escaped:
				// Vorheriges Zeichen war Backslash, dieses Zeichen überspringen
				escaped = false
			case r == '\\':
				// Nächstes Zeichen als escaped markieren
				escaped = true
			case r == '"':
				// Ende des String-Literals
				inString = false
			}
			continue
		}

		switch r {
		case '"':
			// Start eines String-Literals
			inString = true
		case '{', '[':
			// Verschachtelungsebene erhöhen für Objekte und Arrays
			depth++
		case '}', ']':
			// Verschachtelungsebene verringern
			depth--
			if depth == 0 {
				// Ende der Root-JSON-Struktur erreicht
				return i
			}
		}
	}

	return -1
}
