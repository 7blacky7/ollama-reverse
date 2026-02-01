// Package parser - Modelfile-Parser für Ollama
// Modul command: Command-Strukturen und Hilfsfunktionen
package parser

import (
	"fmt"
	"slices"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
)

var deprecatedParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

// CreateRequest creates a new *api.CreateRequest from an existing Modelfile
func (f Modelfile) CreateRequest(relativeDir string) (*api.CreateRequest, error) {
	req := &api.CreateRequest{}

	var messages []api.Message
	var licenses []string
	params := make(map[string]any)

	for _, c := range f.Commands {
		switch c.Name {
		case "model":
			path, err := expandPath(c.Args, relativeDir)
			if err != nil {
				return nil, err
			}

			digestMap, err := fileDigestMap(path)
			if isNotExist(err) {
				req.From = c.Args
				continue
			} else if err != nil {
				return nil, err
			}

			if req.Files == nil {
				req.Files = digestMap
			} else {
				for k, v := range digestMap {
					req.Files[k] = v
				}
			}
		case "adapter":
			path, err := expandPath(c.Args, relativeDir)
			if err != nil {
				return nil, err
			}

			digestMap, err := fileDigestMap(path)
			if err != nil {
				return nil, err
			}

			req.Adapters = digestMap
		case "template":
			req.Template = c.Args
		case "system":
			req.System = c.Args
		case "license":
			licenses = append(licenses, c.Args)
		case "renderer":
			req.Renderer = c.Args
		case "parser":
			req.Parser = c.Args
		case "requires":
			// golang.org/x/mod/semver requires "v" prefix
			requires := c.Args
			if !strings.HasPrefix(requires, "v") {
				requires = "v" + requires
			}
			if !semver.IsValid(requires) {
				return nil, fmt.Errorf("requires must be a valid semver (e.g. 0.14.0)")
			}
			req.Requires = strings.TrimPrefix(requires, "v")
		case "message":
			role, msg, _ := strings.Cut(c.Args, ": ")
			messages = append(messages, api.Message{Role: role, Content: msg})
		default:
			if slices.Contains(deprecatedParameters, c.Name) {
				fmt.Printf("warning: parameter %s is deprecated\n", c.Name)
				break
			}

			ps, err := api.FormatParams(map[string][]string{c.Name: {c.Args}})
			if err != nil {
				return nil, err
			}

			for k, v := range ps {
				if ks, ok := params[k].([]string); ok {
					params[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					params[k] = vs
				} else {
					params[k] = v
				}
			}
		}
	}

	if len(params) > 0 {
		req.Parameters = params
	}
	if len(messages) > 0 {
		req.Messages = messages
	}
	if len(licenses) > 0 {
		req.License = licenses
	}

	return req, nil
}

type Command struct {
	Name string
	Args string
}

func (c Command) String() string {
	var sb strings.Builder
	switch c.Name {
	case "model":
		fmt.Fprintf(&sb, "FROM %s", c.Args)
	case "license", "template", "system", "adapter", "renderer", "parser", "requires":
		fmt.Fprintf(&sb, "%s %s", strings.ToUpper(c.Name), quote(c.Args))
	case "message":
		role, message, _ := strings.Cut(c.Args, ": ")
		fmt.Fprintf(&sb, "MESSAGE %s %s", role, quote(message))
	default:
		fmt.Fprintf(&sb, "PARAMETER %s %s", c.Name, quote(c.Args))
	}

	return sb.String()
}

func quote(s string) string {
	if strings.Contains(s, "\n") || strings.HasPrefix(s, " ") || strings.HasSuffix(s, " ") {
		if strings.Contains(s, "\"") {
			return `"""` + s + `"""`
		}

		return `"` + s + `"`
	}

	return s
}

func unquote(s string) (string, bool) {
	// TODO: single quotes
	if len(s) >= 3 && s[:3] == `"""` {
		if len(s) >= 6 && s[len(s)-3:] == `"""` {
			return s[3 : len(s)-3], true
		}

		return "", false
	}

	if len(s) >= 1 && s[0] == '"' {
		if len(s) >= 2 && s[len(s)-1] == '"' {
			return s[1 : len(s)-1], true
		}

		return "", false
	}

	return s, true
}

func isAlpha(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
}

func isNumber(r rune) bool {
	return r >= '0' && r <= '9'
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t'
}

func isNewline(r rune) bool {
	return r == '\r' || r == '\n'
}

func isValidMessageRole(role string) bool {
	return role == "system" || role == "user" || role == "assistant"
}

func isValidCommand(cmd string) bool {
	switch strings.ToLower(cmd) {
	case "from", "license", "template", "system", "adapter", "renderer", "parser", "parameter", "message", "requires":
		return true
	default:
		return false
	}
}
