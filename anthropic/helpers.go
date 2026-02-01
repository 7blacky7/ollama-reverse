// helpers.go
// Hilfsfunktionen-Modul: ID-Generierung und Utility-Funktionen

package anthropic

import (
	"crypto/rand"
	"fmt"
	"time"

	"github.com/ollama/ollama/api"
)

// generateID generates a unique ID with the given prefix using crypto/rand
func generateID(prefix string) string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		// Fallback to time-based ID if crypto/rand fails
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s_%x", prefix, b)
}

// GenerateMessageID generates a unique message ID
func GenerateMessageID() string {
	return generateID("msg")
}

// ptr returns a pointer to the given string value
func ptr(s string) *string {
	return &s
}

// mapToArgs converts a map to ToolCallFunctionArguments
func mapToArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}
