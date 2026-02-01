// Package template - Template-Engine für Ollama
// Modul nodes: Parse-Tree-Operationen und Node-Traversierung
package template

import (
	"errors"
	"text/template"
	"text/template/parse"
)

// Subtree returns a new template containing only the subtree matching the predicate
func (t *Template) Subtree(fn func(parse.Node) bool) *template.Template {
	var walk func(parse.Node) parse.Node
	walk = func(n parse.Node) parse.Node {
		if fn(n) {
			return n
		}

		switch t := n.(type) {
		case *parse.ListNode:
			for _, c := range t.Nodes {
				if n := walk(c); n != nil {
					return n
				}
			}
		case *parse.BranchNode:
			for _, n := range []*parse.ListNode{t.List, t.ElseList} {
				if n != nil {
					if n := walk(n); n != nil {
						return n
					}
				}
			}
		case *parse.IfNode:
			return walk(&t.BranchNode)
		case *parse.WithNode:
			return walk(&t.BranchNode)
		case *parse.RangeNode:
			return walk(&t.BranchNode)
		}

		return nil
	}

	if n := walk(t.Tree.Root); n != nil {
		return (&template.Template{
			Tree: &parse.Tree{
				Root: &parse.ListNode{
					Nodes: []parse.Node{n},
				},
			},
		}).Funcs(funcs)
	}

	return nil
}

// Identifiers walks the node tree returning any identifiers it finds along the way
func Identifiers(n parse.Node) ([]string, error) {
	switch n := n.(type) {
	case *parse.ListNode:
		var names []string
		for _, n := range n.Nodes {
			i, err := Identifiers(n)
			if err != nil {
				return names, err
			}
			names = append(names, i...)
		}

		return names, nil
	case *parse.TemplateNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined template specified")
		}
		return Identifiers(n.Pipe)
	case *parse.ActionNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined action in template")
		}
		return Identifiers(n.Pipe)
	case *parse.BranchNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined branch")
		}
		names, err := Identifiers(n.Pipe)
		if err != nil {
			return names, err
		}
		for _, n := range []*parse.ListNode{n.List, n.ElseList} {
			if n != nil {
				i, err := Identifiers(n)
				if err != nil {
					return names, err
				}
				names = append(names, i...)
			}
		}
		return names, nil
	case *parse.IfNode:
		return Identifiers(&n.BranchNode)
	case *parse.RangeNode:
		return Identifiers(&n.BranchNode)
	case *parse.WithNode:
		return Identifiers(&n.BranchNode)
	case *parse.PipeNode:
		var names []string
		for _, c := range n.Cmds {
			for _, a := range c.Args {
				i, err := Identifiers(a)
				if err != nil {
					return names, err
				}
				names = append(names, i...)
			}
		}
		return names, nil
	case *parse.FieldNode:
		return n.Ident, nil
	case *parse.VariableNode:
		return n.Ident, nil
	}

	return nil, nil
}

// deleteNode walks the node list and deletes nodes that match the predicate
// this is currently to remove the {{ .Response }} node from templates
func deleteNode(n parse.Node, fn func(parse.Node) bool) parse.Node {
	var walk func(n parse.Node) parse.Node
	walk = func(n parse.Node) parse.Node {
		if fn(n) {
			return nil
		}

		switch t := n.(type) {
		case *parse.ListNode:
			var nodes []parse.Node
			for _, c := range t.Nodes {
				if n := walk(c); n != nil {
					nodes = append(nodes, n)
				}
			}

			t.Nodes = nodes
			return t
		case *parse.IfNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.WithNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.RangeNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.BranchNode:
			t.List = walk(t.List).(*parse.ListNode)
			if t.ElseList != nil {
				t.ElseList = walk(t.ElseList).(*parse.ListNode)
			}
		case *parse.ActionNode:
			n := walk(t.Pipe)
			if n == nil {
				return nil
			}

			t.Pipe = n.(*parse.PipeNode)
		case *parse.PipeNode:
			var commands []*parse.CommandNode
			for _, c := range t.Cmds {
				var args []parse.Node
				for _, a := range c.Args {
					if n := walk(a); n != nil {
						args = append(args, n)
					}
				}

				if len(args) == 0 {
					return nil
				}

				c.Args = args
				commands = append(commands, c)
			}

			if len(commands) == 0 {
				return nil
			}

			t.Cmds = commands
		}

		return n
	}

	return walk(n)
}
