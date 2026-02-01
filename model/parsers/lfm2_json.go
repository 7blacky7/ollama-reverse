// lfm2_json.go - JSON-Parsing fuer LFM2 Tool-Calls
//
// Enthaelt:
// - parseLFM2JSON: Parst JSON-Format Tool-Calls
package parsers

import (
	"encoding/json"

	"github.com/ollama/ollama/api"
)

// parseLFM2JSON tries to parse content as JSON format tool call
// Returns nil, nil if content is not valid JSON format
func parseLFM2JSON(content string) ([]api.ToolCall, error) {
	var parsed struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	if err := json.Unmarshal([]byte(content), &parsed); err == nil && parsed.Name != "" {
		var args api.ToolCallFunctionArguments
		if len(parsed.Arguments) > 0 {
			if err := json.Unmarshal(parsed.Arguments, &args); err != nil {
				return nil, err
			}
		} else {
			args = api.NewToolCallFunctionArguments()
		}

		return []api.ToolCall{{
			Function: api.ToolCallFunction{
				Name:      parsed.Name,
				Arguments: args,
			},
		}}, nil
	}

	return nil, nil
}
