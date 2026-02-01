// Package agent provides agent loop orchestration and tool approval.
// Datei: approval_types.go
// Inhalt: Typen, Konstanten und Lookup-Tabellen fuer Tool-Approval
package agent

import (
	"strings"
)

// ApprovalDecision represents the user's decision for a tool execution.
type ApprovalDecision int

const (
	// ApprovalDeny means the user denied execution.
	ApprovalDeny ApprovalDecision = iota
	// ApprovalOnce means execute this one time only.
	ApprovalOnce
	// ApprovalAlways means add to session allowlist.
	ApprovalAlways
)

// ApprovalResult contains the decision and optional deny reason.
type ApprovalResult struct {
	Decision   ApprovalDecision
	DenyReason string
}

// Option labels for the selector (numbered for quick selection)
var optionLabels = []string{
	"1. Execute once",
	"2. Allow for this session",
	"3. Deny",
}

// toolDisplayNames maps internal tool names to human-readable display names.
var toolDisplayNames = map[string]string{
	"bash":       "Bash",
	"web_search": "Web Search",
	"web_fetch":  "Web Fetch",
}

// ToolDisplayName returns the human-readable display name for a tool.
func ToolDisplayName(toolName string) string {
	if displayName, ok := toolDisplayNames[toolName]; ok {
		return displayName
	}
	// Default: capitalize first letter and replace underscores with spaces
	name := strings.ReplaceAll(toolName, "_", " ")
	if len(name) > 0 {
		return strings.ToUpper(name[:1]) + name[1:]
	}
	return toolName
}

// autoAllowCommands are commands that are always allowed without prompting.
// These are zero-risk, read-only commands.
var autoAllowCommands = map[string]bool{
	"pwd":      true,
	"echo":     true,
	"date":     true,
	"whoami":   true,
	"hostname": true,
	"uname":    true,
}

// autoAllowPrefixes are command prefixes that are always allowed.
// These are read-only or commonly-needed development commands.
var autoAllowPrefixes = []string{
	// Git read-only
	"git status", "git log", "git diff", "git branch", "git show",
	"git remote -v", "git tag", "git stash list",
	// Package managers - run scripts
	"npm run", "npm test", "npm start",
	"bun run", "bun test",
	"uv run",
	"yarn run", "yarn test",
	"pnpm run", "pnpm test",
	// Package info
	"go list", "go version", "go env",
	"npm list", "npm ls", "npm version",
	"pip list", "pip show",
	"cargo tree", "cargo version",
	// Build commands
	"go build", "go test", "go fmt", "go vet",
	"make", "cmake",
	"cargo build", "cargo test", "cargo check",
}

// denyPatterns are dangerous command patterns that are always blocked.
var denyPatterns = []string{
	// Destructive commands
	"rm -rf", "rm -fr",
	"mkfs", "dd if=", "dd of=",
	"shred",
	"> /dev/", ">/dev/",
	// Privilege escalation
	"sudo ", "su ", "doas ",
	"chmod 777", "chmod -R 777",
	"chown ", "chgrp ",
	// Network exfiltration
	"curl -d", "curl --data", "curl -X POST", "curl -X PUT",
	"wget --post",
	"nc ", "netcat ",
	"scp ", "rsync ",
	// History and credentials
	"history",
	".bash_history", ".zsh_history",
	".ssh/id_rsa", ".ssh/id_dsa", ".ssh/id_ecdsa", ".ssh/id_ed25519",
	".ssh/config",
	".aws/credentials", ".aws/config",
	".gnupg/",
	"/etc/shadow", "/etc/passwd",
	// Dangerous patterns
	":(){ :|:& };:", // fork bomb
	"chmod +s",      // setuid
	"mkfifo",
}

// denyPathPatterns are file patterns that should never be accessed.
// These are checked as exact filename matches or path suffixes.
var denyPathPatterns = []string{
	".env",
	".env.local",
	".env.production",
	"credentials.json",
	"secrets.json",
	"secrets.yaml",
	"secrets.yml",
	".pem",
	".key",
}
