// glm46_helpers.go - Hilfsfunktionen fuer GLM-4.6 Parser
//
// Enthaelt:
// - GLMToolCallXML: XML-Struktur fuer Tool-Calls
// - escapeGLM46Content: Escapt XML-Entitaeten
// - parseGLM46ToolCall: Parst Tool-Call aus XML
package parsers

import (
	"encoding/xml"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// GLMToolCallXML represents the structure of a GLM-4.6 tool call for XML parsing
type GLMToolCallXML struct {
	XMLName xml.Name `xml:"tool_call"`
	Content string   `xml:",chardata"` // Function name (text nodes between tags)
	Keys    []string `xml:"arg_key"`   // All arg_key elements in document order
	Values  []string `xml:"arg_value"` // All arg_value elements in document order
}

// escapeGLM46Content escapes XML entities in text content while preserving arg_key/arg_value tags
func escapeGLM46Content(s string) string {
	var result strings.Builder
	inTag := false

	for i := range len(s) {
		ch := s[i]

		if ch == '<' {
			// Check if this is a known tag
			if strings.HasPrefix(s[i:], "<arg_key>") ||
				strings.HasPrefix(s[i:], "</arg_key>") ||
				strings.HasPrefix(s[i:], "<arg_value>") ||
				strings.HasPrefix(s[i:], "</arg_value>") {
				inTag = true
			}
		}

		if inTag {
			result.WriteByte(ch)
			if ch == '>' {
				inTag = false
			}
		} else {
			// Escape special characters in text content
			switch ch {
			case '&':
				result.WriteString("&amp;")
			case '<':
				result.WriteString("&lt;")
			case '>':
				result.WriteString("&gt;")
			default:
				result.WriteByte(ch)
			}
		}
	}

	return result.String()
}

func parseGLM46ToolCall(raw glm46EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	// Escape any unescaped entities in text content
	// We need to escape text between tags, but not the tags themselves
	escaped := escapeGLM46Content(raw.raw)

	// Wrap the content in a root element to make it valid XML
	xmlString := "<tool_call>" + escaped + "</tool_call>"

	// Parse XML into struct
	var parsed GLMToolCallXML
	if err := xml.Unmarshal([]byte(xmlString), &parsed); err != nil {
		return api.ToolCall{}, fmt.Errorf("failed to parse XML: %w", err)
	}

	// Extract and trim function name
	functionName := strings.TrimSpace(parsed.Content)
	if functionName == "" {
		return api.ToolCall{}, fmt.Errorf("empty function name")
	}

	// Verify keys and values are paired correctly
	if len(parsed.Keys) != len(parsed.Values) {
		return api.ToolCall{}, fmt.Errorf("mismatched arg_key and arg_value counts: %d keys, %d values", len(parsed.Keys), len(parsed.Values))
	}

	// Find the matching tool to get parameter types
	var matchedTool *api.Tool
	for i := range tools {
		if tools[i].Function.Name == functionName {
			matchedTool = &tools[i]
			break
		}
	}

	// Build arguments map by pairing keys and values
	toolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      functionName,
			Arguments: api.NewToolCallFunctionArguments(),
		},
	}

	for i := range parsed.Keys {
		key := strings.TrimSpace(parsed.Keys[i])
		value := parsed.Values[i] // Don't trim here - parseValue handles it

		// Look up parameter type
		var paramType api.PropertyType
		if matchedTool != nil && matchedTool.Function.Parameters.Properties != nil {
			if prop, ok := matchedTool.Function.Parameters.Properties.Get(key); ok {
				// Handle anyOf by collecting all types from the union
				if len(prop.AnyOf) > 0 {
					for _, anyOfProp := range prop.AnyOf {
						paramType = append(paramType, anyOfProp.Type...)
					}
				} else {
					paramType = prop.Type
				}
			}
		}

		// Parse value with type coercion
		toolCall.Function.Arguments.Set(key, parseValue(value, paramType))
	}

	return toolCall, nil
}
