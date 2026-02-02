// =============================================================================
// Modul: functiongemma_values.go
// Beschreibung: Wert-Parsing-Funktionen für den FunctionGemma-Parser.
//               Enthält Logik zum Parsen von Argumenten, Arrays und Objekten
//               im FunctionGemma-spezifischen Format (key:value,key:value).
// =============================================================================

package parsers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// parseArguments parst das key:value,key:value Format in eine ToolCallFunctionArguments-Map.
// Die Funktion respektiert verschachtelte Strukturen und <escape>-Tags.
func (p *FunctionGemmaParser) parseArguments(argsStr string) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	if argsStr == "" {
		return args
	}

	// Aufteilen nach Komma, aber verschachtelte Strukturen beachten
	parts := p.splitArguments(argsStr)

	for _, part := range parts {
		// Finde den ersten Doppelpunkt um key:value zu trennen
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			continue
		}

		key := part[:colonIdx]
		value := part[colonIdx+1:]

		// Wert parsen und setzen
		args.Set(key, p.parseValue(value))
	}

	return args
}

// splitArguments teilt Argumente am Komma auf, respektiert dabei verschachtelte Strukturen.
// Verschachtelungen durch {} und [] werden erkannt, ebenso <escape>-Tags.
func (p *FunctionGemmaParser) splitArguments(argsStr string) []string {
	var parts []string
	var current strings.Builder
	depth := 0
	inEscape := false

	for i := 0; i < len(argsStr); i++ {
		ch := argsStr[i]

		// Prüfe auf <escape>-Tags
		if i+8 <= len(argsStr) && argsStr[i:i+8] == "<escape>" {
			inEscape = !inEscape
			current.WriteString("<escape>")
			i += 7 // Überspringe den Rest von <escape>
			continue
		}

		if !inEscape {
			switch ch {
			case '{', '[':
				depth++
				current.WriteByte(ch)
			case '}', ']':
				depth--
				current.WriteByte(ch)
			case ',':
				if depth == 0 {
					if current.Len() > 0 {
						parts = append(parts, current.String())
						current.Reset()
					}
					continue
				}
				current.WriteByte(ch)
			default:
				current.WriteByte(ch)
			}
		} else {
			current.WriteByte(ch)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	return parts
}

// parseValue parst einen einzelnen Wert aus dem FunctionGemma-Format.
// Unterstützt: escaped Strings, Booleans, Zahlen, Arrays und Objekte.
func (p *FunctionGemmaParser) parseValue(value string) any {
	// Prüfe auf escaped String
	if strings.HasPrefix(value, "<escape>") && strings.HasSuffix(value, "<escape>") {
		// Entferne die Escape-Tags
		return value[8 : len(value)-8]
	}

	// Prüfe auf Boolean
	if value == "true" {
		return true
	}
	if value == "false" {
		return false
	}

	// Prüfe auf Zahl
	if num, ok := parseNumber(value); ok {
		return num
	}

	// Prüfe auf Array
	if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		return p.parseArray(value[1 : len(value)-1])
	}

	// Prüfe auf Objekt
	if strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}") {
		return p.parseObject(value[1 : len(value)-1])
	}

	// Standard: String zurückgeben
	return value
}

// parseArray parst einen Array-Wert und gibt ein []any zurück.
func (p *FunctionGemmaParser) parseArray(content string) []any {
	var result []any
	parts := p.splitArguments(content)
	for _, part := range parts {
		result = append(result, p.parseValue(part))
	}
	return result
}

// parseObject parst einen Objekt-Wert und gibt eine map[string]any zurück.
func (p *FunctionGemmaParser) parseObject(content string) map[string]any {
	result := make(map[string]any)
	parts := p.splitArguments(content)
	for _, part := range parts {
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			continue
		}
		key := part[:colonIdx]
		value := part[colonIdx+1:]
		result[key] = p.parseValue(value)
	}
	return result
}

// parseNumber versucht einen String als Zahl zu parsen.
// Gibt (Zahl, true) zurück bei Erfolg, (nil, false) bei Fehlschlag.
func parseNumber(s string) (any, bool) {
	// Versuche zuerst Integer
	var intVal int64
	if _, err := fmt.Sscanf(s, "%d", &intVal); err == nil {
		// Prüfe ob der gesamte String konsumiert wurde
		if fmt.Sprintf("%d", intVal) == s {
			return intVal, true
		}
	}

	// Versuche Float
	var floatVal float64
	if _, err := fmt.Sscanf(s, "%f", &floatVal); err == nil {
		return floatVal, true
	}

	return nil, false
}
