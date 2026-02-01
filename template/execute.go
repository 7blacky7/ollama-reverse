// Package template - Template-Engine für Ollama
// Modul execute: Template-Ausführung und Nachrichten-Kollation
package template

import (
	"bytes"
	"io"
	"slices"
	"strings"
	"text/template"
	"text/template/parse"

	"github.com/ollama/ollama/api"
)

type Values struct {
	Messages []api.Message
	api.Tools
	Prompt string
	Suffix string
	Think  bool
	// ThinkLevel contains the thinking level if Think is true and a string value was provided
	ThinkLevel string
	// whether or not the user explicitly set the thinking flag (vs. it being
	// implicitly false). Templates can't see whether `Think` is nil
	IsThinkSet bool

	// forceLegacy is a flag used to test compatibility with legacy templates
	forceLegacy bool
}

func (t *Template) Execute(w io.Writer, v Values) error {
	system, messages := collate(v.Messages)
	vars, err := t.Vars()
	if err != nil {
		return err
	}
	if v.Prompt != "" && v.Suffix != "" {
		return t.Template.Execute(w, map[string]any{
			"Prompt":     v.Prompt,
			"Suffix":     v.Suffix,
			"Response":   "",
			"Think":      v.Think,
			"ThinkLevel": v.ThinkLevel,
			"IsThinkSet": v.IsThinkSet,
		})
	} else if !v.forceLegacy && slices.Contains(vars, "messages") {
		return t.Template.Execute(w, map[string]any{
			"System":     system,
			"Messages":   convertMessagesForTemplate(messages),
			"Tools":      convertToolsForTemplate(v.Tools),
			"Response":   "",
			"Think":      v.Think,
			"ThinkLevel": v.ThinkLevel,
			"IsThinkSet": v.IsThinkSet,
		})
	}

	system = ""
	var b bytes.Buffer
	var prompt, responseStr string
	for _, m := range messages {
		execute := func() error {
			if err := t.Template.Execute(&b, map[string]any{
				"System":     system,
				"Prompt":     prompt,
				"Response":   responseStr,
				"Think":      v.Think,
				"ThinkLevel": v.ThinkLevel,
				"IsThinkSet": v.IsThinkSet,
			}); err != nil {
				return err
			}

			system = ""
			prompt = ""
			responseStr = ""
			return nil
		}

		switch m.Role {
		case "system":
			if prompt != "" || responseStr != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			system = m.Content
		case "user":
			if responseStr != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			prompt = m.Content
		case "assistant":
			responseStr = m.Content
		}
	}

	var cut bool
	nodes := deleteNode(t.Template.Root.Copy(), func(n parse.Node) bool {
		if field, ok := n.(*parse.FieldNode); ok && slices.Contains(field.Ident, "Response") {
			cut = true
			return false
		}

		return cut
	})

	tree := parse.Tree{Root: nodes.(*parse.ListNode)}
	if err := template.Must(template.New("").AddParseTree("", &tree)).Execute(&b, map[string]any{
		"System":     system,
		"Prompt":     prompt,
		"Response":   responseStr,
		"Think":      v.Think,
		"ThinkLevel": v.ThinkLevel,
		"IsThinkSet": v.IsThinkSet,
	}); err != nil {
		return err
	}

	_, err = io.Copy(w, &b)
	return err
}

// collate messages based on role. consecutive messages of the same role are merged
// into a single message (except for tool messages which preserve individual metadata).
// collate also collects and returns all system messages.
// collate mutates message content adding image tags ([img-%d]) as needed
// todo(parthsareen): revisit for contextual image support
func collate(msgs []api.Message) (string, []*api.Message) {
	var system []string
	var collated []*api.Message
	for i := range msgs {
		if msgs[i].Role == "system" {
			system = append(system, msgs[i].Content)
		}

		// merges consecutive messages of the same role into a single message (except for tool messages)
		if len(collated) > 0 && collated[len(collated)-1].Role == msgs[i].Role && msgs[i].Role != "tool" {
			collated[len(collated)-1].Content += "\n\n" + msgs[i].Content
		} else {
			collated = append(collated, &msgs[i])
		}
	}

	return strings.Join(system, "\n\n"), collated
}
