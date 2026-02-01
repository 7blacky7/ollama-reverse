// Package template - Template-Engine für Ollama
// Hauptmodul: Template-Parsing und Grundstrukturen
package template

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"
	"text/template"
	"text/template/parse"
	"time"

	"github.com/agnivade/levenshtein"

	"github.com/ollama/ollama/api"
)

//go:embed index.json
var indexBytes []byte

//go:embed *.gotmpl
//go:embed *.json
var templatesFS embed.FS

var templatesOnce = sync.OnceValues(func() ([]*named, error) {
	var templates []*named
	if err := json.Unmarshal(indexBytes, &templates); err != nil {
		return nil, err
	}

	for _, t := range templates {
		bts, err := templatesFS.ReadFile(t.Name + ".gotmpl")
		if err != nil {
			return nil, err
		}

		// normalize line endings
		t.Bytes = bytes.ReplaceAll(bts, []byte("\r\n"), []byte("\n"))

		params, err := templatesFS.ReadFile(t.Name + ".json")
		if err != nil {
			continue
		}

		if err := json.Unmarshal(params, &t.Parameters); err != nil {
			return nil, err
		}
	}

	return templates, nil
})

type named struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Bytes    []byte

	Parameters *struct {
		Stop []string `json:"stop"`
	}
}

func (t named) Reader() io.Reader {
	return bytes.NewReader(t.Bytes)
}

func Named(s string) (*named, error) {
	templates, err := templatesOnce()
	if err != nil {
		return nil, err
	}

	var template *named
	score := math.MaxInt
	for _, t := range templates {
		if s := levenshtein.ComputeDistance(s, t.Template); s < score {
			score = s
			template = t
		}
	}

	if score < 100 {
		return template, nil
	}

	return nil, errors.New("no matching template found")
}

var DefaultTemplate, _ = Parse("{{ .Prompt }}")

type Template struct {
	*template.Template
	raw string
}

// response is a template node that can be added to templates that don't already have one
var response = parse.ActionNode{
	NodeType: parse.NodeAction,
	Pipe: &parse.PipeNode{
		NodeType: parse.NodePipe,
		Cmds: []*parse.CommandNode{
			{
				NodeType: parse.NodeCommand,
				Args: []parse.Node{
					&parse.FieldNode{
						NodeType: parse.NodeField,
						Ident:    []string{"Response"},
					},
				},
			},
		},
	},
}

var funcs = template.FuncMap{
	"json": func(v any) string {
		b, _ := json.Marshal(v)
		return string(b)
	},
	"currentDate": func(args ...string) string {
		// Currently ignoring the format argument, but accepting it for future use
		// Default format is YYYY-MM-DD
		return time.Now().Format("2006-01-02")
	},
	"yesterdayDate": func(args ...string) string {
		return time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	},
	"toTypeScriptType": func(v any) string {
		if param, ok := v.(api.ToolProperty); ok {
			return param.ToTypeScriptType()
		}
		// Handle pointer case
		if param, ok := v.(*api.ToolProperty); ok && param != nil {
			return param.ToTypeScriptType()
		}
		return "any"
	},
}

func Parse(s string) (*Template, error) {
	tmpl := template.New("").Option("missingkey=zero").Funcs(funcs)

	tmpl, err := tmpl.Parse(s)
	if err != nil {
		return nil, err
	}

	t := Template{Template: tmpl, raw: s}
	vars, err := t.Vars()
	if err != nil {
		return nil, err
	}

	if !slices.Contains(vars, "messages") && !slices.Contains(vars, "response") {
		// touch up the template and append {{ .Response }}
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, &response)
	}

	return &t, nil
}

func (t *Template) String() string {
	return t.raw
}

func (t *Template) Vars() ([]string, error) {
	var vars []string
	for _, tt := range t.Templates() {
		for _, n := range tt.Root.Nodes {
			v, err := Identifiers(n)
			if err != nil {
				return vars, err
			}
			vars = append(vars, v...)
		}
	}

	set := make(map[string]struct{})
	for _, n := range vars {
		set[strings.ToLower(n)] = struct{}{}
	}

	return slices.Sorted(maps.Keys(set)), nil
}

func (t *Template) Contains(s string) bool {
	return strings.Contains(t.raw, s)
}
