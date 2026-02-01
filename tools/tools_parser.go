// tools_parser.go - Parser für Tool-Aufrufe
// Enthält: Parser-Struktur, State-Management und Add-Logik

package tools

import (
	"bytes"
	"strings"
	"text/template"

	"github.com/ollama/ollama/api"
)

type toolsState int

const (
	toolsState_LookingForTag toolsState = iota
	toolsState_ToolCalling
	toolsState_Done
)

type Parser struct {
	tag   string
	tools []api.Tool

	state  toolsState
	buffer []byte
	n      int
}

func (p *Parser) GetBuffer() []byte {
	return p.buffer
}

// NewParser creates a new tool call parser from a model's chat
// template and a list of provided tools.
func NewParser(tmpl *template.Template, tools []api.Tool) *Parser {
	return NewParserWithTag(tools, parseTag(tmpl))
}

func NewParserWithTag(tools []api.Tool, tag string) *Parser {
	return &Parser{
		tag:   tag,
		tools: tools,
	}
}

// Add processes a string input to parse tool calls and content that
// should be sent back to the user.
func (p *Parser) Add(s string) (calls []api.ToolCall, content string) {
	if p.state == toolsState_Done {
		return nil, s
	}

	p.buffer = append(p.buffer, s...)

	if p.state == toolsState_LookingForTag {
		i, found := p.findTag()
		if i == -1 {
			content = string(p.buffer)
			p.buffer = []byte{}
		} else {
			content = string(p.buffer[:i])
			p.buffer = p.buffer[i:]
		}

		// for models where { or [ are used as tool calling
		// tags, we only support parsing tools if the first non-
		// whitespace character is { or [
		if p.tag == "{" || p.tag == "[" {
			if strings.TrimSpace(content) != "" {
				p.state = toolsState_Done
				return nil, content + string(p.buffer)
			}
		}

		if !found {
			return nil, content
		}

		p.state = toolsState_ToolCalling
	}

	for {
		call := p.parseToolCall()
		if call == nil {
			break
		}

		calls = append(calls, *call)
	}

	if p.done() {
		p.state = toolsState_Done
		content = string(p.buffer)
		p.buffer = []byte{}
	}

	return calls, content
}

// findTag searches the buffer to find and handle a tool calling tag
// returning true if the tag was found and false otherwise, and
// a string content signaling any content that should be sent back to the user
func (p *Parser) findTag() (int, bool) {
	// First check for complete substring anywhere in s
	if i := bytes.Index(p.buffer, []byte(p.tag)); i > -1 {
		return i, true
	}

	// Then check for partial suffix overlap
	max := min(len(p.buffer), len(p.tag))
	for i := max; i > 0; i-- {
		if bytes.HasSuffix(p.buffer, []byte(p.tag[:i])) {
			return len(p.buffer) - i, false
		}
	}
	return -1, false
}

// parseToolCall finds the next complete tool call in the buffer
// incrementing n and advancing the buffer.
func (p *Parser) parseToolCall() *api.ToolCall {
	tool, end := findTool(p.tools, p.buffer)
	if tool == nil {
		return nil
	}

	var argsMap map[string]any
	if found, i := findArguments(tool, p.buffer); found == nil {
		return nil
	} else {
		argsMap = found
		if i > end {
			end = i
		}
	}

	args := api.NewToolCallFunctionArguments()
	for k, v := range argsMap {
		args.Set(k, v)
	}

	tc := &api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      tool.Function.Name,
			Arguments: args,
			Index:     p.n,
		},
	}

	p.n++
	p.buffer = p.buffer[end:]
	return tc
}

// done checks if the parser is done parsing by looking
// for closing tag. currently only } and ] are supported
// for closing tags as {} or [] pairs may not always
// represent tool calls and we need to send the content back
func (p *Parser) done() bool {
	var open, close rune
	switch p.tag {
	case "{":
		open, close = '{', '}'
	case "[":
		open, close = '[', ']'
	default:
		return false
	}

	var count int
	for _, c := range p.buffer {
		if c == byte(open) {
			count++
		} else if c == byte(close) {
			count--
			if count == 0 {
				return true
			}
		}
	}

	return false
}

// Content returns any remaining content that
// should be sent to the user. This should be the empty string
// string unless the tag is { or [ and a tool call was not found
func (p *Parser) Content() string {
	if p.n > 0 {
		return ""
	}

	if p.tag == "{" || p.tag == "[" {
		return string(p.buffer)
	}

	return ""
}
