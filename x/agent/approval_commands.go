// Package agent provides agent loop orchestration and tool approval.
// Datei: approval_commands.go
// Inhalt: Hilfsfunktionen fuer Bash-Command-Analyse (Prefix-Extraktion, Deny-Check, CWD-Check)
package agent

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// IsAutoAllowed checks if a bash command is auto-allowed (no prompt needed).
func IsAutoAllowed(command string) bool {
	command = strings.TrimSpace(command)

	// Check exact command match (first word)
	fields := strings.Fields(command)
	if len(fields) > 0 && autoAllowCommands[fields[0]] {
		return true
	}

	// Check prefix match
	for _, prefix := range autoAllowPrefixes {
		if strings.HasPrefix(command, prefix) {
			return true
		}
	}

	return false
}

// IsDenied checks if a bash command matches deny patterns.
// Returns true and the matched pattern if denied.
func IsDenied(command string) (bool, string) {
	commandLower := strings.ToLower(command)

	// Check deny patterns
	for _, pattern := range denyPatterns {
		if strings.Contains(commandLower, strings.ToLower(pattern)) {
			return true, pattern
		}
	}

	// Check deny path patterns
	for _, pattern := range denyPathPatterns {
		if strings.Contains(commandLower, strings.ToLower(pattern)) {
			return true, pattern
		}
	}

	return false, ""
}

// FormatDeniedResult returns the tool result message when a command is blocked.
func FormatDeniedResult(command string, pattern string) string {
	return fmt.Sprintf("Command blocked: this command matches a dangerous pattern (%s) and cannot be executed. If this command is necessary, please ask the user to run it manually.", pattern)
}

// extractBashPrefix extracts a prefix pattern from a bash command.
// For commands like "cat tools/tools_test.go | head -200", returns "cat:tools/"
// For commands without path args, returns empty string.
// Paths with ".." traversal that escape the base directory return empty string for security.
func extractBashPrefix(command string) string {
	// Split command by pipes and get the first part
	parts := strings.Split(command, "|")
	firstCmd := strings.TrimSpace(parts[0])

	// Split into command and args
	fields := strings.Fields(firstCmd)
	if len(fields) < 2 {
		return ""
	}

	baseCmd := fields[0]
	// Common commands that benefit from prefix allowlisting
	// These are typically safe for read operations on specific directories
	safeCommands := map[string]bool{
		"cat": true, "ls": true, "head": true, "tail": true,
		"less": true, "more": true, "file": true, "wc": true,
		"grep": true, "find": true, "tree": true, "stat": true,
		"sed": true,
	}

	if !safeCommands[baseCmd] {
		return ""
	}

	// Find the first path-like argument (must contain / or \ or start with .)
	// First pass: look for clear paths (containing path separators or starting with .)
	for _, arg := range fields[1:] {
		// Skip flags
		if strings.HasPrefix(arg, "-") {
			continue
		}
		// Skip numeric arguments (e.g., "head -n 100")
		if isNumeric(arg) {
			continue
		}
		// Only process if it looks like a path (contains / or \ or starts with .)
		if !strings.Contains(arg, "/") && !strings.Contains(arg, "\\") && !strings.HasPrefix(arg, ".") {
			continue
		}
		// Normalize to forward slashes for consistent cross-platform matching
		arg = strings.ReplaceAll(arg, "\\", "/")

		// Security: reject absolute paths
		if path.IsAbs(arg) {
			return "" // Absolute path - don't create prefix
		}

		// Normalize the path using stdlib path.Clean (resolves . and ..)
		cleaned := path.Clean(arg)

		// Security: reject if cleaned path escapes to parent directory
		if strings.HasPrefix(cleaned, "..") {
			return "" // Path escapes - don't create prefix
		}

		// Security: if original had "..", verify cleaned path didn't escape to sibling
		// e.g., "tools/a/b/../../../etc" -> "etc" (escaped tools/ to sibling)
		if strings.Contains(arg, "..") {
			origBase := strings.SplitN(arg, "/", 2)[0]
			cleanedBase := strings.SplitN(cleaned, "/", 2)[0]
			if origBase != cleanedBase {
				return "" // Path escaped to sibling directory
			}
		}

		// Check if arg ends with / (explicit directory)
		isDir := strings.HasSuffix(arg, "/")

		// Get the directory part
		var dir string
		if isDir {
			dir = cleaned
		} else {
			dir = path.Dir(cleaned)
		}

		if dir == "." {
			return fmt.Sprintf("%s:./", baseCmd)
		}
		return fmt.Sprintf("%s:%s/", baseCmd, dir)
	}

	// Second pass: if no clear path found, use the first non-flag argument as a filename
	for _, arg := range fields[1:] {
		if strings.HasPrefix(arg, "-") {
			continue
		}
		if isNumeric(arg) {
			continue
		}
		// Treat as filename in current dir
		return fmt.Sprintf("%s:./", baseCmd)
	}

	return ""
}

// isNumeric checks if a string is a numeric value
func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// isCommandOutsideCwd checks if a bash command targets paths outside the current working directory.
// Returns true if any path argument would access files outside cwd.
func isCommandOutsideCwd(command string) bool {
	cwd, err := os.Getwd()
	if err != nil {
		return false // Can't determine, assume safe
	}

	// Split command by pipes and semicolons to check all parts
	parts := strings.FieldsFunc(command, func(r rune) bool {
		return r == '|' || r == ';' || r == '&'
	})

	for _, part := range parts {
		part = strings.TrimSpace(part)
		fields := strings.Fields(part)
		if len(fields) == 0 {
			continue
		}

		// Check each argument that looks like a path
		for _, arg := range fields[1:] {
			// Skip flags
			if strings.HasPrefix(arg, "-") {
				continue
			}

			// Treat POSIX-style absolute paths as outside cwd on all platforms.
			if strings.HasPrefix(arg, "/") || strings.HasPrefix(arg, "\\") {
				return true
			}

			// Check for absolute paths outside cwd
			if filepath.IsAbs(arg) {
				absPath := filepath.Clean(arg)
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
				continue
			}

			// Check for relative paths that escape cwd (e.g., ../foo, /etc/passwd)
			if strings.HasPrefix(arg, "..") {
				// Resolve the path relative to cwd
				absPath := filepath.Join(cwd, arg)
				absPath = filepath.Clean(absPath)
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
			}

			// Check for home directory expansion
			if strings.HasPrefix(arg, "~") {
				home, err := os.UserHomeDir()
				if err == nil && !strings.HasPrefix(home, cwd) {
					return true
				}
			}
		}
	}

	return false
}

// AllowlistKey generates the key for exact allowlist lookup.
func AllowlistKey(toolName string, args map[string]any) string {
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			return fmt.Sprintf("bash:%s", cmd)
		}
	}
	return toolName
}
