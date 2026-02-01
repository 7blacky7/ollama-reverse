// Package model - Validierung und interne Hilfsfunktionen
// Enthaelt: Validierungslogik, partKind-Pruefungen, String-Hilfsfunktionen
package model

import (
	"cmp"
	"path/filepath"
	"strings"
)

// IsValidNamespace reports whether the provided string is a valid
// namespace.
func IsValidNamespace(s string) bool {
	return isValidPart(kindNamespace, s)
}

// IsValid reports whether all parts of the name are present and valid. The
// digest is a special case, and is checked for validity only if present.
//
// Note: The digest check has been removed as is planned to be added back in
// at a later time.
func (n Name) IsValid() bool {
	return n.IsFullyQualified()
}

// IsFullyQualified returns true if all parts of the name are present and
// valid without the digest.
func (n Name) IsFullyQualified() bool {
	parts := []string{
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	}
	for i, part := range parts {
		if !isValidPart(partKind(i), part) {
			return false
		}
	}
	return true
}

// Filepath returns a canonical filepath that represents the name with each part from
// host to tag as a directory in the form:
//
//	{host}/{namespace}/{model}/{tag}
//
// It uses the system's filepath separator and ensures the path is clean.
//
// It panics if the name is not fully qualified. Use [Name.IsFullyQualified]
// to check if the name is fully qualified.
func (n Name) Filepath() string {
	if !n.IsFullyQualified() {
		panic("illegal attempt to get filepath of invalid name")
	}
	return filepath.Join(
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	)
}

// isValidLen checks if the string length is valid for the given part kind.
func isValidLen(kind partKind, s string) bool {
	switch kind {
	case kindHost:
		return len(s) >= 1 && len(s) <= 350
	case kindTag:
		return len(s) >= 1 && len(s) <= 80
	default:
		return len(s) >= 1 && len(s) <= 80
	}
}

// isValidPart validates a single part of the name based on its kind.
func isValidPart(kind partKind, s string) bool {
	if !isValidLen(kind, s) {
		return false
	}
	for i := range s {
		if i == 0 {
			if !isAlphanumericOrUnderscore(s[i]) {
				return false
			}
			continue
		}
		switch s[i] {
		case '_', '-':
		case '.':
			if kind == kindNamespace {
				return false
			}
		case ':':
			if kind != kindHost && kind != kindDigest {
				return false
			}
		default:
			if !isAlphanumericOrUnderscore(s[i]) {
				return false
			}
		}
	}
	return true
}

// isAlphanumericOrUnderscore checks if a byte is alphanumeric or underscore.
func isAlphanumericOrUnderscore(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_'
}

// cutLast cuts the string at the last occurrence of the separator.
func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// cutPromised cuts the last part of s at the last occurrence of sep. If sep is
// found, the part before and after sep are returned as-is unless empty, in
// which case they are returned as MissingPart, which will cause
// [Name.IsValid] to return false.
func cutPromised(s, sep string) (before, after string, ok bool) {
	before, after, ok = cutLast(s, sep)
	if !ok {
		return before, after, false
	}
	return cmp.Or(before, MissingPart), cmp.Or(after, MissingPart), true
}
