// olmo3_helpers.go - Hilfsfunktionen fuer Olmo3 Parser
//
// Enthaelt:
// - parseOlmo3FunctionCalls: Parst Python-Style Funktionsaufrufe
// - parseOlmo3SingleFunctionCall: Parst einzelnen Funktionsaufruf
// - parseOlmo3Arguments: Parst Argumente mit Verschachtelung
// - splitArguments: Trennt Argumente respektiert Quotes
// - parseOlmo3Value: Parst Werte (String, Number, Bool, Array, Object)
package parsers

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
)

// parseOlmo3FunctionCalls parses function calls in Python-esque format:
// func_name(arg1="value1", arg2=123)
// Multiple calls are separated by newlines
func parseOlmo3FunctionCalls(s string) ([]api.ToolCall, error) {
	var calls []api.ToolCall
	s = strings.TrimSpace(s)
	if s == "" {
		return calls, nil
	}

	// Split by newlines for multiple function calls
	lines := strings.Split(s, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		call, err := parseOlmo3SingleFunctionCall(line)
		if err != nil {
			return nil, fmt.Errorf("failed to parse function call %q: %w", line, err)
		}
		calls = append(calls, call)
	}

	return calls, nil
}

// Regex to match function call: func_name(args)
var funcCallRegex = regexp.MustCompile(`^(\w+)\((.*)\)$`)

func parseOlmo3SingleFunctionCall(s string) (api.ToolCall, error) {
	matches := funcCallRegex.FindStringSubmatch(s)
	if matches == nil {
		return api.ToolCall{}, fmt.Errorf("invalid function call format")
	}

	funcName := matches[1]
	argsStr := matches[2]

	args, err := parseOlmo3Arguments(argsStr)
	if err != nil {
		return api.ToolCall{}, fmt.Errorf("failed to parse arguments: %w", err)
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      funcName,
			Arguments: args,
		},
	}, nil
}

// parseOlmo3Arguments parses comma-separated key=value pairs
// Handles nested parentheses, brackets, braces, and quoted strings
func parseOlmo3Arguments(s string) (api.ToolCallFunctionArguments, error) {
	args := api.NewToolCallFunctionArguments()
	s = strings.TrimSpace(s)
	if s == "" {
		return args, nil
	}

	// Split by commas, but respect nested structures and quotes
	parts := splitArguments(s)

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Find the first = sign
		eqIdx := strings.Index(part, "=")
		if eqIdx == -1 {
			return api.ToolCallFunctionArguments{}, fmt.Errorf("invalid argument format: %s", part)
		}

		key := strings.TrimSpace(part[:eqIdx])
		valueStr := strings.TrimSpace(part[eqIdx+1:])

		value, err := parseOlmo3Value(valueStr)
		if err != nil {
			return api.ToolCallFunctionArguments{}, fmt.Errorf("failed to parse value for %s: %w", key, err)
		}

		args.Set(key, value)
	}

	return args, nil
}

// splitArguments splits arguments by commas, respecting quotes and nested structures
func splitArguments(s string) []string {
	var parts []string
	var current strings.Builder
	depth := 0
	inString := false
	stringChar := byte(0)
	escaped := false

	for i := range s {
		c := s[i]

		if escaped {
			current.WriteByte(c)
			escaped = false
			continue
		}

		if c == '\\' && inString {
			current.WriteByte(c)
			escaped = true
			continue
		}

		if (c == '"' || c == '\'') && !inString {
			inString = true
			stringChar = c
			current.WriteByte(c)
			continue
		}

		if c == stringChar && inString {
			inString = false
			stringChar = 0
			current.WriteByte(c)
			continue
		}

		if !inString {
			switch c {
			case '(', '[', '{':
				depth++
				current.WriteByte(c)
			case ')', ']', '}':
				depth--
				current.WriteByte(c)
			case ',':
				if depth == 0 {
					parts = append(parts, current.String())
					current.Reset()
					continue
				}
				current.WriteByte(c)
			default:
				current.WriteByte(c)
			}
		} else {
			current.WriteByte(c)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	return parts
}

// parseOlmo3Value parses a value which can be a string, number, boolean, null, array, or object
func parseOlmo3Value(s string) (any, error) {
	s = strings.TrimSpace(s)

	// Check for quoted string
	if (strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`)) ||
		(strings.HasPrefix(s, `'`) && strings.HasSuffix(s, `'`)) {
		// Remove quotes and unescape
		inner := s[1 : len(s)-1]
		return unescapeString(inner), nil
	}

	// Check for boolean
	if s == "true" || s == "True" {
		return true, nil
	}
	if s == "false" || s == "False" {
		return false, nil
	}

	// Check for null/None
	if s == "null" || s == "None" || s == "nil" {
		return nil, nil
	}

	// Check for number
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f, nil
	}

	// Check for array [...]
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		return parseOlmo3Array(s[1 : len(s)-1])
	}

	// Check for object {...}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		return parseOlmo3Object(s[1 : len(s)-1])
	}

	// Default to string without quotes
	return s, nil
}

func parseOlmo3Array(s string) ([]any, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []any{}, nil
	}

	parts := splitArguments(s)
	var arr []any
	for _, part := range parts {
		val, err := parseOlmo3Value(part)
		if err != nil {
			return nil, err
		}
		arr = append(arr, val)
	}
	return arr, nil
}

func parseOlmo3Object(s string) (map[string]any, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return map[string]any{}, nil
	}

	// Objects use key: value or "key": value format
	obj := make(map[string]any)
	parts := splitArguments(s)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Find colon separator
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			return nil, fmt.Errorf("invalid object entry: %s", part)
		}

		keyStr := strings.TrimSpace(part[:colonIdx])
		valueStr := strings.TrimSpace(part[colonIdx+1:])

		// Remove quotes from key if present
		if (strings.HasPrefix(keyStr, `"`) && strings.HasSuffix(keyStr, `"`)) ||
			(strings.HasPrefix(keyStr, `'`) && strings.HasSuffix(keyStr, `'`)) {
			keyStr = keyStr[1 : len(keyStr)-1]
		}

		val, err := parseOlmo3Value(valueStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse value for key %s: %w", keyStr, err)
		}

		obj[keyStr] = val
	}

	return obj, nil
}

func unescapeString(s string) string {
	// Handle common escape sequences
	s = strings.ReplaceAll(s, `\\`, "\x00") // Placeholder for backslash
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\'`, `'`)
	s = strings.ReplaceAll(s, `\n`, "\n")
	s = strings.ReplaceAll(s, `\t`, "\t")
	s = strings.ReplaceAll(s, `\r`, "\r")
	s = strings.ReplaceAll(s, "\x00", `\`) // Restore backslash
	return s
}
