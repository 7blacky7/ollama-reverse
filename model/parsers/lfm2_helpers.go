// lfm2_helpers.go - Hilfsfunktionen fuer LFM2 Tool-Call-Parsing
//
// Enthaelt:
// - parseToolCallsContent: Parst ein oder mehrere Tool-Calls aus Content
// - parsePythonStyleToolCalls: Parst Python-Style Funktionsaufrufe
// - findMatchingParen: Findet schliessende Klammer (mit Verschachtelung)
// - parsePythonArgs: Parst Python-Keyword-Argumente
package parsers

import (
	"errors"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
)

// parseToolCallsContent parses one or more tool calls from content
// Supports JSON format and Python-style format including multiple calls: [func1(...),func2(...)]
func (p *LFM2Parser) parseToolCallsContent(content string) ([]api.ToolCall, error) {
	content = strings.TrimSpace(content)

	// Try JSON format first: {"name": "func", "arguments": {...}}
	toolCalls, err := parseLFM2JSON(content)
	if err == nil && len(toolCalls) > 0 {
		return toolCalls, nil
	}

	// Try Python-style format: [func(arg1='val1'),func2(arg2='val2')] or func(arg1='val1')
	return p.parsePythonStyleToolCalls(content)
}

// parsePythonStyleToolCalls parses one or more Python-style tool calls
// Examples: [bash(command='ls'),bash(command='pwd')] or bash(command='ls')
func (p *LFM2Parser) parsePythonStyleToolCalls(content string) ([]api.ToolCall, error) {
	content = strings.TrimSpace(content)

	// Strip outer brackets if present: [func(...)] -> func(...)
	if strings.HasPrefix(content, "[") && strings.HasSuffix(content, "]") {
		content = content[1 : len(content)-1]
	}

	var toolCalls []api.ToolCall

	// Parse multiple function calls separated by commas at the top level
	for len(content) > 0 {
		content = strings.TrimSpace(content)
		if content == "" {
			break
		}

		// Skip leading comma from previous iteration
		if strings.HasPrefix(content, ",") {
			content = strings.TrimSpace(content[1:])
			if content == "" {
				break
			}
		}

		// Find function name
		parenIdx := strings.Index(content, "(")
		if parenIdx == -1 {
			return nil, errors.New("invalid tool call: no opening parenthesis")
		}

		funcName := strings.TrimSpace(content[:parenIdx])
		if funcName == "" {
			return nil, errors.New("invalid tool call: empty function name")
		}

		// Find matching closing parenthesis
		closeIdx := findMatchingParen(content, parenIdx)
		if closeIdx == -1 {
			return nil, errors.New("invalid tool call: no matching closing parenthesis")
		}

		argsStr := content[parenIdx+1 : closeIdx]
		args := api.NewToolCallFunctionArguments()

		if argsStr != "" {
			if err := parsePythonArgs(argsStr, &args); err != nil {
				return nil, err
			}
		}

		toolCalls = append(toolCalls, api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      funcName,
				Arguments: args,
			},
		})

		// Move past this function call
		content = content[closeIdx+1:]
	}

	if len(toolCalls) == 0 {
		return nil, errors.New("no tool calls found")
	}

	return toolCalls, nil
}

// findMatchingParen finds the index of the closing parenthesis matching the one at openIdx
// Returns -1 if not found. Handles nested parentheses and quoted strings.
func findMatchingParen(s string, openIdx int) int {
	depth := 1
	i := openIdx + 1
	for i < len(s) && depth > 0 {
		switch s[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i
			}
		case '\'', '"':
			// Skip quoted string
			quote := s[i]
			i++
			for i < len(s) && s[i] != quote {
				if s[i] == '\\' && i+1 < len(s) {
					i++ // skip escaped char
				}
				i++
			}
		}
		i++
	}
	return -1
}

// parseToolCallContent parses a single tool call (for backward compatibility with tests)
func (p *LFM2Parser) parseToolCallContent(content string) (api.ToolCall, error) {
	calls, err := p.parseToolCallsContent(content)
	if err != nil {
		return api.ToolCall{}, err
	}
	if len(calls) == 0 {
		return api.ToolCall{}, errors.New("no tool call found")
	}
	return calls[0], nil
}

// parsePythonArgs parses Python-style keyword arguments: key='value', key2="value2"
func parsePythonArgs(argsStr string, args *api.ToolCallFunctionArguments) error {
	// Simple state machine to parse key='value' pairs
	// Handles: command='ls', flag="-la", count=42, enabled=true
	var key string
	i := 0

	for i < len(argsStr) {
		// Skip whitespace
		for i < len(argsStr) && (argsStr[i] == ' ' || argsStr[i] == '\t' || argsStr[i] == '\n') {
			i++
		}
		if i >= len(argsStr) {
			break
		}

		// Parse key
		keyStart := i
		for i < len(argsStr) && argsStr[i] != '=' && argsStr[i] != ',' {
			i++
		}
		if i >= len(argsStr) || argsStr[i] != '=' {
			return errors.New("invalid argument: expected '='")
		}
		key = strings.TrimSpace(argsStr[keyStart:i])
		i++ // skip '='

		// Skip whitespace after =
		for i < len(argsStr) && (argsStr[i] == ' ' || argsStr[i] == '\t') {
			i++
		}

		// Parse value
		var value string
		if i < len(argsStr) && (argsStr[i] == '\'' || argsStr[i] == '"') {
			// Quoted string
			quote := argsStr[i]
			i++
			valueStart := i
			for i < len(argsStr) && argsStr[i] != quote {
				if argsStr[i] == '\\' && i+1 < len(argsStr) {
					i += 2 // skip escaped char
				} else {
					i++
				}
			}
			value = argsStr[valueStart:i]
			if i < len(argsStr) {
				i++ // skip closing quote
			}
			args.Set(key, value)
		} else {
			// Unquoted value (number, bool, etc)
			valueStart := i
			for i < len(argsStr) && argsStr[i] != ',' {
				i++
			}
			value = strings.TrimSpace(argsStr[valueStart:i])

			// Try to parse as number or bool
			if v, err := strconv.ParseInt(value, 10, 64); err == nil {
				args.Set(key, v)
			} else if v, err := strconv.ParseFloat(value, 64); err == nil {
				args.Set(key, v)
			} else if value == "true" {
				args.Set(key, true)
			} else if value == "false" {
				args.Set(key, false)
			} else {
				args.Set(key, value)
			}
		}

		// Skip comma and whitespace
		for i < len(argsStr) && (argsStr[i] == ',' || argsStr[i] == ' ' || argsStr[i] == '\t' || argsStr[i] == '\n') {
			i++
		}
	}

	return nil
}
