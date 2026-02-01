//go:build ignore

// generate_types.go - Typdefinitionen und Hilfsfunktionen fuer MLX-Wrapper-Generator
//
// Enthaelt:
// - Function Struct Definition
// - Header-Suche (findHeaders)
// - Content-Bereinigung (cleanContent)
// - Parameter-Extraktion (extractParamNames, splitParams)
// - ARM64-Guard-Erkennung (needsARM64Guard)
package main

import (
	"bytes"
	"io/fs"
	"path/filepath"
	"regexp"
	"strings"
)

// Function repraesentiert eine geparste C-Funktion
type Function struct {
	Name            string
	ReturnType      string
	Params          string
	ParamNames      []string
	NeedsARM64Guard bool
}

// findHeaders sucht alle .h Dateien im angegebenen Verzeichnis
func findHeaders(directory string) ([]string, error) {
	var headers []string
	err := filepath.WalkDir(directory, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".h") {
			headers = append(headers, path)
		}
		return nil
	})
	return headers, err
}

// cleanContent entfernt Kommentare und Praeprozessor-Direktiven
func cleanContent(content string) string {
	// Remove single-line comments
	re := regexp.MustCompile(`//.*?\n`)
	content = re.ReplaceAllString(content, "\n")

	// Remove multi-line comments
	re = regexp.MustCompile(`/\*.*?\*/`)
	content = re.ReplaceAllString(content, "")

	// Remove preprocessor directives (lines starting with #) - use multiline mode
	re = regexp.MustCompile(`(?m)^\s*#.*?$`)
	content = re.ReplaceAllString(content, "")

	// Remove extern "C" { and } blocks more conservatively
	// Only remove the extern "C" { line, not the content inside
	re = regexp.MustCompile(`extern\s+"C"\s*\{\s*?\n`)
	content = re.ReplaceAllString(content, "\n")
	// Remove standalone closing braces that are not part of function declarations
	re = regexp.MustCompile(`\n\s*\}\s*\n`)
	content = re.ReplaceAllString(content, "\n")

	// Collapse whitespace and newlines
	re = regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	return content
}

// extractParamNames extrahiert Parameternamen aus Funktionsparametern
func extractParamNames(params string) []string {
	if params == "" || strings.TrimSpace(params) == "void" {
		return []string{}
	}

	var names []string

	// Split by comma, but respect parentheses (for function pointers)
	parts := splitParams(params)

	// Remove array brackets
	arrayBrackets := regexp.MustCompile(`\[.*?\]`)

	// Function pointer pattern
	funcPtrPattern := regexp.MustCompile(`\(\s*\*\s*(\w+)\s*\)`)

	// Type keywords to skip
	typeKeywords := map[string]bool{
		"const":     true,
		"struct":    true,
		"unsigned":  true,
		"signed":    true,
		"long":      true,
		"short":     true,
		"int":       true,
		"char":      true,
		"float":     true,
		"double":    true,
		"void":      true,
		"size_t":    true,
		"uint8_t":   true,
		"uint16_t":  true,
		"uint32_t":  true,
		"uint64_t":  true,
		"int8_t":    true,
		"int16_t":   true,
		"int32_t":   true,
		"int64_t":   true,
		"intptr_t":  true,
		"uintptr_t": true,
	}

	for _, part := range parts {
		if part == "" {
			continue
		}

		// Remove array brackets
		part = arrayBrackets.ReplaceAllString(part, "")

		// For function pointers like "void (*callback)(int)"
		if matches := funcPtrPattern.FindStringSubmatch(part); len(matches) > 1 {
			names = append(names, matches[1])
			continue
		}

		// Regular parameter: last identifier
		tokens := regexp.MustCompile(`\w+`).FindAllString(part, -1)
		if len(tokens) > 0 {
			// The last token is usually the parameter name
			// Skip type keywords
			for i := len(tokens) - 1; i >= 0; i-- {
				if !typeKeywords[tokens[i]] {
					names = append(names, tokens[i])
					break
				}
			}
		}
	}

	return names
}

// splitParams teilt Parameter unter Beachtung von Klammern
func splitParams(params string) []string {
	var parts []string
	var current bytes.Buffer
	depth := 0

	for _, char := range params + "," {
		switch char {
		case '(':
			depth++
			current.WriteRune(char)
		case ')':
			depth--
			current.WriteRune(char)
		case ',':
			if depth == 0 {
				parts = append(parts, strings.TrimSpace(current.String()))
				current.Reset()
			} else {
				current.WriteRune(char)
			}
		default:
			current.WriteRune(char)
		}
	}

	return parts
}

// needsARM64Guard prueft ob ARM64-spezifische Guards benoetigt werden
func needsARM64Guard(name, retType, params string) bool {
	return strings.Contains(name, "float16") ||
		strings.Contains(name, "bfloat16") ||
		strings.Contains(retType, "float16_t") ||
		strings.Contains(retType, "bfloat16_t") ||
		strings.Contains(params, "float16_t") ||
		strings.Contains(params, "bfloat16_t")
}
