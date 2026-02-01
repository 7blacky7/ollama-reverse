// types_tools.go - Tool Types fuer Function Calling
// Enthaelt: Tools, Tool, ToolCall, ToolCallFunction, ToolCallFunctionArguments,
//           ToolFunction, ToolFunctionParameters, ToolPropertiesMap, ToolProperty, PropertyType
package api

import (
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/ollama/ollama/internal/orderedmap"
)

// ============================================================================
// Tool Call Types (fuer Responses)
// ============================================================================

type Tools []Tool

func (t Tools) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Index     int                       `json:"index"`
	Name      string                    `json:"name"`
	Arguments ToolCallFunctionArguments `json:"arguments"`
}

// ToolCallFunctionArguments holds tool call arguments in insertion order.
type ToolCallFunctionArguments struct {
	om *orderedmap.Map[string, any]
}

// NewToolCallFunctionArguments creates a new empty ToolCallFunctionArguments.
func NewToolCallFunctionArguments() ToolCallFunctionArguments {
	return ToolCallFunctionArguments{om: orderedmap.New[string, any]()}
}

// Get retrieves a value by key.
func (t *ToolCallFunctionArguments) Get(key string) (any, bool) {
	if t == nil || t.om == nil {
		return nil, false
	}
	return t.om.Get(key)
}

// Set sets a key-value pair, preserving insertion order.
func (t *ToolCallFunctionArguments) Set(key string, value any) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, any]()
	}
	t.om.Set(key, value)
}

// Len returns the number of arguments.
func (t *ToolCallFunctionArguments) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all key-value pairs in insertion order.
func (t *ToolCallFunctionArguments) All() iter.Seq2[string, any] {
	if t == nil || t.om == nil {
		return func(yield func(string, any) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolCallFunctionArguments) ToMap() map[string]any {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t *ToolCallFunctionArguments) String() string {
	if t == nil || t.om == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t.om)
	return string(bts)
}

func (t *ToolCallFunctionArguments) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, any]()
	return json.Unmarshal(data, t.om)
}

func (t ToolCallFunctionArguments) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(t.om)
}

// ============================================================================
// Tool Definition Types (fuer Requests)
// ============================================================================

type Tool struct {
	Type     string       `json:"type"`
	Items    any          `json:"items,omitempty"`
	Function ToolFunction `json:"function"`
}

func (t Tool) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  ToolFunctionParameters `json:"parameters"`
}

func (t *ToolFunction) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

type ToolFunctionParameters struct {
	Type       string             `json:"type"`
	Defs       any                `json:"$defs,omitempty"`
	Items      any                `json:"items,omitempty"`
	Required   []string           `json:"required,omitempty"`
	Properties *ToolPropertiesMap `json:"properties"`
}

func (t *ToolFunctionParameters) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// ============================================================================
// Tool Property Types
// ============================================================================

// PropertyType can be either a string or an array of strings
type PropertyType []string

// UnmarshalJSON implements the json.Unmarshaler interface
func (pt *PropertyType) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*pt = []string{s}
		return nil
	}

	var a []string
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*pt = a
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (pt PropertyType) MarshalJSON() ([]byte, error) {
	if len(pt) == 1 {
		return json.Marshal(pt[0])
	}
	return json.Marshal([]string(pt))
}

// String returns a string representation of the PropertyType
func (pt PropertyType) String() string {
	if len(pt) == 0 {
		return ""
	}
	if len(pt) == 1 {
		return pt[0]
	}
	return fmt.Sprintf("%v", []string(pt))
}

// ToolPropertiesMap holds tool properties in insertion order.
type ToolPropertiesMap struct {
	om *orderedmap.Map[string, ToolProperty]
}

// NewToolPropertiesMap creates a new empty ToolPropertiesMap.
func NewToolPropertiesMap() *ToolPropertiesMap {
	return &ToolPropertiesMap{om: orderedmap.New[string, ToolProperty]()}
}

// Get retrieves a property by name.
func (t *ToolPropertiesMap) Get(key string) (ToolProperty, bool) {
	if t == nil || t.om == nil {
		return ToolProperty{}, false
	}
	return t.om.Get(key)
}

// Set sets a property, preserving insertion order.
func (t *ToolPropertiesMap) Set(key string, value ToolProperty) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, ToolProperty]()
	}
	t.om.Set(key, value)
}

// Len returns the number of properties.
func (t *ToolPropertiesMap) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all properties in insertion order.
func (t *ToolPropertiesMap) All() iter.Seq2[string, ToolProperty] {
	if t == nil || t.om == nil {
		return func(yield func(string, ToolProperty) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolPropertiesMap) ToMap() map[string]ToolProperty {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t ToolPropertiesMap) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.om)
}

func (t *ToolPropertiesMap) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, ToolProperty]()
	return json.Unmarshal(data, t.om)
}

type ToolProperty struct {
	AnyOf       []ToolProperty     `json:"anyOf,omitempty"`
	Type        PropertyType       `json:"type,omitempty"`
	Items       any                `json:"items,omitempty"`
	Description string             `json:"description,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	Properties  *ToolPropertiesMap `json:"properties,omitempty"`
}

// ToTypeScriptType converts a ToolProperty to a TypeScript type string
func (tp ToolProperty) ToTypeScriptType() string {
	if len(tp.AnyOf) > 0 {
		var types []string
		for _, anyOf := range tp.AnyOf {
			types = append(types, anyOf.ToTypeScriptType())
		}
		return strings.Join(types, " | ")
	}

	if len(tp.Type) == 0 {
		return "any"
	}

	if len(tp.Type) == 1 {
		return mapToTypeScriptType(tp.Type[0])
	}

	var types []string
	for _, t := range tp.Type {
		types = append(types, mapToTypeScriptType(t))
	}
	return strings.Join(types, " | ")
}

// mapToTypeScriptType maps JSON Schema types to TypeScript types
func mapToTypeScriptType(jsonType string) string {
	switch jsonType {
	case "string":
		return "string"
	case "number", "integer":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		return "any[]"
	case "object":
		return "Record<string, any>"
	case "null":
		return "null"
	default:
		return "any"
	}
}
