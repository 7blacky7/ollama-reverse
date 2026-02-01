// Package harmony implementiert den Harmony-Parser fuer Ollama.
//
// Modul: harmony_tools.go - ToolCallAccumulator und FunctionNameMap
// Enthaelt: HarmonyToolCallAccumulator, FunctionNameMap und Namensmapping
package harmony

import (
	"fmt"
	"log/slog"
	"strings"
	"unicode"
)

type harmonyToolCallState int

const (
	harmonyToolCallState_Normal harmonyToolCallState = iota
	harmonyToolCallState_ToolCalling
)

type HarmonyToolCallAccumulator struct {
	state           harmonyToolCallState
	acc             strings.Builder
	currentToolName *string
}

func (a *HarmonyToolCallAccumulator) SetToolName(toolName string) {
	a.currentToolName = &toolName
}

func (a *HarmonyToolCallAccumulator) Add(content string) {
	a.acc.WriteString(content)
}

func (a *HarmonyToolCallAccumulator) Drain() (*string, string) {
	str := a.acc.String()
	a.state = harmonyToolCallState_Normal
	a.acc.Reset()
	return a.currentToolName, str
}

func (a *HarmonyToolCallAccumulator) Content() string {
	return a.acc.String()
}

// FunctionNameMap maps a user-specified function name to a valid function
// name for harmony (which look like TypeScript identifiers). This is needed to
// transform user-specified function names, which might contain characters that
// are not allowed in TypeScript identifiers
type FunctionNameMap struct {
	userToHarmony map[string]string
	harmonyToUser map[string]string
}

func NewFunctionNameMap() *FunctionNameMap {
	return &FunctionNameMap{
		userToHarmony: make(map[string]string),
		harmonyToUser: make(map[string]string),
	}
}

func (m *FunctionNameMap) ConvertAndAdd(userFunctionName string) string {
	harmonyFunctionName := m.deriveName(userFunctionName)
	// built-in functions should not be renamed
	if userFunctionName == "browser.open" || userFunctionName == "browser.search" || userFunctionName == "browser.find" || userFunctionName == "python" {
		harmonyFunctionName = userFunctionName
	}
	m.userToHarmony[userFunctionName] = harmonyFunctionName
	m.harmonyToUser[harmonyFunctionName] = userFunctionName
	return harmonyFunctionName
}

// OriginalFromConverted looks up the reverse-mapping of a previously-converted
// user->harmony function name. To unmap reliably, the mapping must exist, as
// the conversion process is not reversible without the appropriate state
func (m *FunctionNameMap) OriginalFromConverted(harmonyFunctionName string) string {
	if userFunctionName, ok := m.harmonyToUser[harmonyFunctionName]; ok {
		return userFunctionName
	}
	slog.Warn("harmony parser: no reverse mapping found for function name", "harmonyFunctionName", harmonyFunctionName)
	// fallback to the original function name if we can't find a mapping
	return harmonyFunctionName
}

// convertToValidChars converts a user-specified function name to a valid
// TypeScript identifier.
//
// Limitations:
//
//   - This doesn't restrict reserved TypeScript keywords.
//   - We don't perform a real ID_Start/ID_Continue check, and instead use the more
//     restrictive unicode.IsLetter/unicode.IsDigit check. Unclear what kind of
//     identifiers these models were trained on, so in the end we might want to
//     convert unicode-heavy identifiers to their closest ASCII equivalents.
func (m *FunctionNameMap) convertToValidChars(userFunctionName string) string {
	mapper := func(r rune) rune {
		// first, replace certain characters with underscores
		if r == ' ' || r == '-' || r == '.' {
			return '_'
		}

		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '$' {
			return r
		}

		// finally, remove any other characters
		return -1
	}
	candidate := strings.Map(mapper, userFunctionName)

	// set a default name if we end up with nothing left
	if candidate == "" {
		return "unnamed"
	}

	// if the candidate starts with a number, prepend an underscore to make it a
	// valid identifier
	if unicode.IsDigit(rune(candidate[0])) {
		candidate = "_" + candidate
	}

	return candidate
}

func (m *FunctionNameMap) deriveName(userFunctionName string) string {
	originalCandidate := m.convertToValidChars(userFunctionName)
	candidate := originalCandidate

	// Check for dupes, and if so, add a number to the end.
	// We start at 2 because if we have dupes and the first is never renamed, it
	// makes sense for them to be named, say, `f`, `f_2`, `f_3`
	count := 2
	for {
		if _, exists := m.harmonyToUser[candidate]; !exists {
			break
		}
		candidate = fmt.Sprintf("%s_%d", originalCandidate, count)
		count++
	}

	return candidate
}
