// Package template - Template-Engine für Ollama
// Modul types: Datentypen für Template-Konvertierung
package template

import (
	"encoding/json"

	"github.com/ollama/ollama/api"
)

// templateTools is a slice of templateTool that marshals to JSON.
type templateTools []templateTool

func (t templateTools) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateArgs is a map type with JSON string output for templates.
type templateArgs map[string]any

func (t templateArgs) String() string {
	if t == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateProperties is a map type with JSON string output for templates.
type templateProperties map[string]api.ToolProperty

func (t templateProperties) String() string {
	if t == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateTool is a template-compatible representation of api.Tool
// with Properties as a regular map for template ranging.
type templateTool struct {
	Type     string               `json:"type"`
	Items    any                  `json:"items,omitempty"`
	Function templateToolFunction `json:"function"`
}

type templateToolFunction struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	Parameters  templateToolFunctionParameters `json:"parameters"`
}

type templateToolFunctionParameters struct {
	Type       string             `json:"type"`
	Defs       any                `json:"$defs,omitempty"`
	Items      any                `json:"items,omitempty"`
	Required   []string           `json:"required,omitempty"`
	Properties templateProperties `json:"properties"`
}

// templateToolCall is a template-compatible representation of api.ToolCall
// with Arguments as a regular map for template ranging.
type templateToolCall struct {
	ID       string
	Function templateToolCallFunction
}

type templateToolCallFunction struct {
	Index     int
	Name      string
	Arguments templateArgs
}

// templateMessage is a template-compatible representation of api.Message
// with ToolCalls converted for template use.
type templateMessage struct {
	Role       string
	Content    string
	Thinking   string
	Images     []api.ImageData
	ToolCalls  []templateToolCall
	ToolName   string
	ToolCallID string
}

// convertToolsForTemplate converts Tools to template-compatible format.
func convertToolsForTemplate(tools api.Tools) templateTools {
	if tools == nil {
		return nil
	}
	result := make(templateTools, len(tools))
	for i, tool := range tools {
		result[i] = templateTool{
			Type:  tool.Type,
			Items: tool.Items,
			Function: templateToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters: templateToolFunctionParameters{
					Type:       tool.Function.Parameters.Type,
					Defs:       tool.Function.Parameters.Defs,
					Items:      tool.Function.Parameters.Items,
					Required:   tool.Function.Parameters.Required,
					Properties: templateProperties(tool.Function.Parameters.Properties.ToMap()),
				},
			},
		}
	}
	return result
}

// convertMessagesForTemplate converts Messages to template-compatible format.
func convertMessagesForTemplate(messages []*api.Message) []*templateMessage {
	if messages == nil {
		return nil
	}
	result := make([]*templateMessage, len(messages))
	for i, msg := range messages {
		var toolCalls []templateToolCall
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, templateToolCall{
				ID: tc.ID,
				Function: templateToolCallFunction{
					Index:     tc.Function.Index,
					Name:      tc.Function.Name,
					Arguments: templateArgs(tc.Function.Arguments.ToMap()),
				},
			})
		}
		result[i] = &templateMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			Thinking:   msg.Thinking,
			Images:     msg.Images,
			ToolCalls:  toolCalls,
			ToolName:   msg.ToolName,
			ToolCallID: msg.ToolCallID,
		}
	}
	return result
}
