// tools_search.go - Such-Funktionen für Tools und Argumente
// Enthält: findTool, findArguments für Tool-Erkennung im Buffer

package tools

import (
	"bytes"
	"encoding/json"

	"github.com/ollama/ollama/api"
)

// findTool finds the first tool name in the list that matches the
// beginning of the buffer, returning nil if no tool is found
// or if the buffer ends with a partial tool name since we need
// to wait for more data to disambiguate.
// The second return value is the end position of the tool name
// if one is found, otherwise 0.
func findTool(tools []api.Tool, buf []byte) (*api.Tool, int) {
	if len(buf) == 0 {
		return nil, 0
	}

	// check if buffer ends with a partial tool name
	// this prevents matching "get" when seeing "get_weather"
	var longest string
	for _, t := range tools {
		if len(t.Function.Name) > len(longest) {
			longest = t.Function.Name
		}
	}

	// Only check up to longest characters from the end
	for i := 1; i <= min(len(buf), len(longest)); i++ {
		tail := buf[len(buf)-i:]
		for _, t := range tools {
			name := []byte(t.Function.Name)
			if len(tail) < len(name) && bytes.HasPrefix(name, tail) {
				return nil, 0
			}
		}
	}

	// find first occurrence of the longest tool name
	var found *api.Tool
	start := -1
	end := -1

	for i := range tools {
		name := []byte(tools[i].Function.Name)
		pos := bytes.Index(buf, name)
		if pos == -1 {
			continue
		}

		// Skip if we have a better match already
		if start != -1 {
			if pos > start {
				continue
			}
			if pos == start && len(name) <= len(found.Function.Name) {
				continue
			}
		}

		found = &tools[i]
		start = pos
		end = pos + len(name)
	}

	if found != nil {
		return found, end
	}

	return nil, 0
}

// findArguments returns the first object that appears to be
// arguments for the provided tool in the provided buffer,
// returning nil if no arguments are found and the end position
// TODO (jmorganca): this does not support parsing omitted arguments
// objects for functions that have all-optional parameters
// e.g. `{"name": "get_conditions", "arguments": {}}` will work but
// `{"name": "get_conditions"}` will not currently work
func findArguments(tool *api.Tool, buffer []byte) (map[string]any, int) {
	if len(buffer) == 0 {
		return nil, 0
	}

	start := -1
	var braces int
	var inString, escaped bool

	for i := range buffer {
		c := buffer[i]

		if escaped {
			escaped = false
			continue
		}

		if c == '\\' {
			escaped = true
			continue
		}

		if c == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if c == '{' {
			if braces == 0 {
				start = i
			}
			braces++
		} else if c == '}' {
			braces--
			if braces == 0 && start != -1 {
				object := buffer[start : i+1]

				var data map[string]any
				if err := json.Unmarshal(object, &data); err != nil {
					// not a valid object, keep looking
					start = -1
					continue
				}

				if args, found := findObject(tool, data); found {
					return args, i
				}

				return data, i
			}

			if braces < 0 {
				braces = 0
			}
		}
	}

	return nil, 0
}

// findObject recursively searches for arguments in a parsed JSON object
func findObject(tool *api.Tool, obj map[string]any) (map[string]any, bool) {
	findMap := func(name string, obj map[string]any) (map[string]any, bool) {
		if args, ok := obj[name].(map[string]any); ok {
			return args, true
		}
		if argsStr, ok := obj[name].(string); ok {
			var argsData map[string]interface{}
			if err := json.Unmarshal([]byte(argsStr), &argsData); err == nil {
				return argsData, ok
			}
		}
		return nil, false
	}

	if _, hasName := obj["name"]; hasName {
		if args, ok := findMap("arguments", obj); ok {
			return args, true
		}
		if args, ok := findMap("parameters", obj); ok {
			return args, true
		}
		return nil, true
	}

	if args, ok := findMap(tool.Function.Name, obj); ok {
		return args, true
	}

	for _, v := range obj {
		switch child := v.(type) {
		case map[string]any:
			if result, found := findObject(tool, child); found {
				return result, true
			}
		case []any:
			for _, item := range child {
				if childObj, ok := item.(map[string]any); ok {
					if result, found := findObject(tool, childObj); found {
						return result, true
					}
				}
			}
		}
	}

	return nil, false
}
