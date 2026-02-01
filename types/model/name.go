// Package model - Kernmodul fuer Model-Namen und Parsing
// Enthaelt: Name-Struktur, Parsing-Funktionen, Konstanten
package model

import (
	"cmp"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"path/filepath"
	"strings"
)

// Errors
var (
	// ErrUnqualifiedName represents an error where a name is not fully
	// qualified. It is not used directly in this package, but is here
	// to avoid other packages inventing their own error type.
	// Additionally, it can be conveniently used via [Unqualified].
	ErrUnqualifiedName = errors.New("unqualified name")
)

// Unqualified is a helper function that returns an error with
// ErrUnqualifiedName as the cause and the name as the message.
func Unqualified(n Name) error {
	return fmt.Errorf("%w: %s", ErrUnqualifiedName, n)
}

// MissingPart is used to indicate any part of a name that was "promised" by
// the presence of a separator, but is missing.
//
// The value was chosen because it is deemed unlikely to be set by a user,
// not a valid part name valid when checked by [Name.IsValid], and easy to
// spot in logs.
const MissingPart = "!MISSING!"

const (
	defaultHost           = "registry.ollama.ai"
	defaultNamespace      = "library"
	defaultTag            = "latest"
	defaultProtocolScheme = "https"
)

// DefaultName returns a name with the default values for the host, namespace,
// tag, and protocol scheme parts. The model and digest parts are empty.
//
//   - The default host is ("registry.ollama.ai")
//   - The default namespace is ("library")
//   - The default tag is ("latest")
//   - The default protocol scheme is ("https")
func DefaultName() Name {
	return Name{
		Host:           defaultHost,
		Namespace:      defaultNamespace,
		Tag:            defaultTag,
		ProtocolScheme: defaultProtocolScheme,
	}
}

type partKind int

const (
	kindHost partKind = iota
	kindNamespace
	kindModel
	kindTag
	kindDigest
)

func (k partKind) String() string {
	switch k {
	case kindHost:
		return "host"
	case kindNamespace:
		return "namespace"
	case kindModel:
		return "model"
	case kindTag:
		return "tag"
	case kindDigest:
		return "digest"
	default:
		return "unknown"
	}
}

// Name is a structured representation of a model name string, as defined by
// [ParseNameNoDefaults].
//
// It is not guaranteed to be valid. Use [Name.IsValid] to check if the name
// is valid.
type Name struct {
	Host           string
	Namespace      string
	Model          string
	Tag            string
	ProtocolScheme string
}

// ParseName parses and assembles a Name from a name string. The
// format of a valid name string is:
//
//	  s:
//		  { host } "/" { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { host } "/" { namespace } "/" { model } ":" { tag }
//		  { host } "/" { namespace } "/" { model } "@" { digest }
//		  { host } "/" { namespace } "/" { model }
//		  { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { namespace } "/" { model } ":" { tag }
//		  { namespace } "/" { model } "@" { digest }
//		  { namespace } "/" { model }
//		  { model } ":" { tag } "@" { digest }
//		  { model } ":" { tag }
//		  { model } "@" { digest }
//		  { model }
//		  "@" { digest }
//	  host:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" }*
//	      length:  [1, 80]
//	  model:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  tag:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  digest:
//	      pattern: { alphanum | "_" } { alphanum | "-" | ":" }*
//	      length:  [1, 80]
//
// Most users should use [ParseName] instead, unless need to support
// different defaults than DefaultName.
//
// The name returned is not guaranteed to be valid. If it is not valid, the
// field values are left in an undefined state. Use [Name.IsValid] to check
// if the name is valid.
func ParseName(s string) Name {
	return Merge(ParseNameBare(s), DefaultName())
}

// ParseNameBare parses s as a name string and returns a Name. No merge with
// [DefaultName] is performed.
func ParseNameBare(s string) Name {
	var n Name
	var promised bool

	// "/" is an illegal tag character, so we can use it to split the host
	if strings.LastIndex(s, ":") > strings.LastIndex(s, "/") {
		s, n.Tag, _ = cutPromised(s, ":")
	}

	s, n.Model, promised = cutPromised(s, "/")
	if !promised {
		n.Model = s
		return n
	}

	s, n.Namespace, promised = cutPromised(s, "/")
	if !promised {
		n.Namespace = s
		return n
	}

	scheme, host, ok := strings.Cut(s, "://")
	if ok {
		n.ProtocolScheme = scheme
	} else {
		host = scheme
	}
	n.Host = host

	return n
}

// ParseNameFromFilepath parses a 4-part filepath as a Name. The parts are
// expected to be in the form:
//
// { host } "/" { namespace } "/" { model } "/" { tag }
func ParseNameFromFilepath(s string) (n Name) {
	parts := strings.Split(s, string(filepath.Separator))
	if len(parts) != 4 {
		return Name{}
	}

	n.Host = parts[0]
	n.Namespace = parts[1]
	n.Model = parts[2]
	n.Tag = parts[3]
	if !n.IsFullyQualified() {
		return Name{}
	}

	return n
}

// Merge merges the host, namespace, tag, and protocol scheme parts of the two names,
// preferring the non-empty parts of a.
func Merge(a, b Name) Name {
	a.Host = cmp.Or(a.Host, b.Host)
	a.Namespace = cmp.Or(a.Namespace, b.Namespace)
	a.Tag = cmp.Or(a.Tag, b.Tag)
	a.ProtocolScheme = cmp.Or(a.ProtocolScheme, b.ProtocolScheme)
	return a
}

// LogValue returns a slog.Value that represents the name as a string.
func (n Name) LogValue() slog.Value {
	return slog.StringValue(n.String())
}

// EqualFold reports whether names are equal under Unicode case-folding.
func (n Name) EqualFold(o Name) bool {
	return strings.EqualFold(n.Host, o.Host) &&
		strings.EqualFold(n.Namespace, o.Namespace) &&
		strings.EqualFold(n.Model, o.Model) &&
		strings.EqualFold(n.Tag, o.Tag)
}

// BaseURL returns the base URL for the registry.
func (n Name) BaseURL() *url.URL {
	return &url.URL{
		Scheme: n.ProtocolScheme,
		Host:   n.Host,
	}
}
