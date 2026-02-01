// Package model - Display- und String-Methoden fuer Name
// Enthaelt: String(), DisplayShortest(), DisplayNamespaceModel()
package model

import (
	"strings"
)

// String returns the name string, in the format that [ParseNameNoDefaults]
// accepts as valid, if [Name.IsValid] reports true; otherwise the empty
// string is returned.
func (n Name) String() string {
	var b strings.Builder
	if n.Host != "" {
		b.WriteString(n.Host)
		b.WriteByte('/')
	}
	if n.Namespace != "" {
		b.WriteString(n.Namespace)
		b.WriteByte('/')
	}
	b.WriteString(n.Model)
	if n.Tag != "" {
		b.WriteByte(':')
		b.WriteString(n.Tag)
	}
	return b.String()
}

// DisplayShortest returns a short string version of the name.
func (n Name) DisplayShortest() string {
	var sb strings.Builder

	if !strings.EqualFold(n.Host, defaultHost) {
		sb.WriteString(n.Host)
		sb.WriteByte('/')
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	} else if !strings.EqualFold(n.Namespace, defaultNamespace) {
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	}

	// always include model and tag
	sb.WriteString(n.Model)
	sb.WriteString(":")
	sb.WriteString(n.Tag)
	return sb.String()
}

// DisplayNamespaceModel returns the namespace and model joined by "/".
func (n Name) DisplayNamespaceModel() string {
	var b strings.Builder
	b.WriteString(n.Namespace)
	b.WriteByte('/')
	b.WriteString(n.Model)
	return b.String()
}
