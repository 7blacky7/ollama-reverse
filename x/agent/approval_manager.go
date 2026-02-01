// Package agent provides agent loop orchestration and tool approval.
// Datei: approval_manager.go
// Inhalt: ApprovalManager-Struct mit Allowlist-Verwaltung und Approval-Request
package agent

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"golang.org/x/term"
)

// ApprovalManager manages tool execution approvals.
type ApprovalManager struct {
	allowlist map[string]bool // exact matches
	prefixes  map[string]bool // prefix matches for bash commands (e.g., "cat:tools/")
	mu        sync.RWMutex
}

// NewApprovalManager creates a new approval manager.
func NewApprovalManager() *ApprovalManager {
	return &ApprovalManager{
		allowlist: make(map[string]bool),
		prefixes:  make(map[string]bool),
	}
}

// IsAllowed checks if a tool/command is allowed (exact match or prefix match).
// For bash commands, hierarchical path matching is used - if "cat:tools/" is allowed,
// then "cat:tools/subdir/" is also allowed (subdirectories inherit parent permissions).
func (a *ApprovalManager) IsAllowed(toolName string, args map[string]any) bool {
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Check exact match first
	key := AllowlistKey(toolName, args)
	if a.allowlist[key] {
		return true
	}

	// For bash commands, check prefix matches with hierarchical path support
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			prefix := extractBashPrefix(cmd)
			if prefix != "" {
				// Check exact prefix match first
				if a.prefixes[prefix] {
					return true
				}
				// Check hierarchical match: if any stored prefix is a parent of current prefix
				// e.g., stored "cat:tools/" should match current "cat:tools/subdir/"
				if a.matchesHierarchicalPrefix(prefix) {
					return true
				}
			}
		}
	}

	// Check if tool itself is allowed (non-bash)
	if toolName != "bash" && a.allowlist[toolName] {
		return true
	}

	return false
}

// matchesHierarchicalPrefix checks if the given prefix matches any stored prefix hierarchically.
// For example, if "cat:tools/" is stored, it will match "cat:tools/subdir/" or "cat:tools/a/b/c/".
func (a *ApprovalManager) matchesHierarchicalPrefix(currentPrefix string) bool {
	// Split prefix into command and path parts (format: "cmd:path/")
	colonIdx := strings.Index(currentPrefix, ":")
	if colonIdx == -1 {
		return false
	}
	currentCmd := currentPrefix[:colonIdx]
	currentPath := currentPrefix[colonIdx+1:]

	for storedPrefix := range a.prefixes {
		storedColonIdx := strings.Index(storedPrefix, ":")
		if storedColonIdx == -1 {
			continue
		}
		storedCmd := storedPrefix[:storedColonIdx]
		storedPath := storedPrefix[storedColonIdx+1:]

		// Commands must match exactly
		if currentCmd != storedCmd {
			continue
		}

		// Check if current path starts with stored path (hierarchical match)
		// e.g., "tools/subdir/" starts with "tools/"
		if strings.HasPrefix(currentPath, storedPath) {
			return true
		}
	}

	return false
}

// AddToAllowlist adds a tool/command to the session allowlist.
// For bash commands, it adds the prefix pattern instead of exact command.
func (a *ApprovalManager) AddToAllowlist(toolName string, args map[string]any) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			prefix := extractBashPrefix(cmd)
			if prefix != "" {
				a.prefixes[prefix] = true
				return
			}
			// Fall back to exact match if no prefix extracted
			a.allowlist[fmt.Sprintf("bash:%s", cmd)] = true
			return
		}
	}
	a.allowlist[toolName] = true
}

// RequestApproval prompts the user for approval to execute a tool.
// Returns the decision and optional deny reason.
func (a *ApprovalManager) RequestApproval(toolName string, args map[string]any) (ApprovalResult, error) {
	// Format tool info for display
	toolDisplay := formatToolDisplay(toolName, args)

	// Enter raw mode for interactive selection
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		// Fallback to simple input if terminal control fails
		return a.fallbackApproval(toolDisplay)
	}

	// Flush any pending stdin input before starting selector
	// This prevents buffered input from causing double-press issues
	flushStdin(fd)

	isWarning := false
	var warningMsg string
	var allowlistInfo string
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			if isCommandOutsideCwd(cmd) {
				isWarning = true
				warningMsg = "command targets paths outside project"
			}
			if prefix := extractBashPrefix(cmd); prefix != "" {
				colonIdx := strings.Index(prefix, ":")
				if colonIdx != -1 {
					cmdName := prefix[:colonIdx]
					dirPath := prefix[colonIdx+1:]
					if dirPath != "./" {
						allowlistInfo = fmt.Sprintf("%s in %s directory (includes subdirs)", cmdName, dirPath)
					} else {
						allowlistInfo = fmt.Sprintf("%s in %s directory", cmdName, dirPath)
					}
				}
			}
		}
	}

	// Run interactive selector
	selected, denyReason, err := runSelector(fd, oldState, toolDisplay, isWarning, warningMsg, allowlistInfo)
	if err != nil {
		term.Restore(fd, oldState)
		return ApprovalResult{Decision: ApprovalDeny}, err
	}

	// Restore terminal
	term.Restore(fd, oldState)

	// Map selection to decision
	switch selected {
	case -1: // Ctrl+C cancelled
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: "cancelled"}, nil
	case 0:
		return ApprovalResult{Decision: ApprovalOnce}, nil
	case 1:
		return ApprovalResult{Decision: ApprovalAlways}, nil
	default:
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: denyReason}, nil
	}
}

// fallbackApproval handles approval when terminal control isn't available.
func (a *ApprovalManager) fallbackApproval(toolDisplay string) (ApprovalResult, error) {
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, toolDisplay)
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "[1] Execute once  [2] Allow for this session  [3] Deny")
	fmt.Fprint(os.Stderr, "choice: ")

	var input string
	fmt.Scanln(&input)

	switch input {
	case "1":
		return ApprovalResult{Decision: ApprovalOnce}, nil
	case "2":
		return ApprovalResult{Decision: ApprovalAlways}, nil
	default:
		fmt.Fprint(os.Stderr, "Reason (optional): ")
		var reason string
		fmt.Scanln(&reason)
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: reason}, nil
	}
}

// Reset clears the session allowlist.
func (a *ApprovalManager) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.allowlist = make(map[string]bool)
	a.prefixes = make(map[string]bool)
}

// AllowedTools returns a list of tools and prefixes in the allowlist.
func (a *ApprovalManager) AllowedTools() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	tools := make([]string, 0, len(a.allowlist)+len(a.prefixes))
	for tool := range a.allowlist {
		tools = append(tools, tool)
	}
	for prefix := range a.prefixes {
		tools = append(tools, prefix+"*")
	}
	return tools
}

// FormatApprovalResult returns a formatted string showing the approval result.
func FormatApprovalResult(toolName string, args map[string]any, result ApprovalResult) string {
	var label string
	displayName := ToolDisplayName(toolName)

	switch result.Decision {
	case ApprovalOnce:
		label = "Approved"
	case ApprovalAlways:
		label = "Always allowed"
	case ApprovalDeny:
		label = "Denied"
	}

	// Format based on tool type
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			// Truncate long commands
			if len(cmd) > 40 {
				cmd = cmd[:37] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, cmd)
		}
	}

	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			// Truncate long queries
			if len(query) > 40 {
				query = query[:37] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, query)
		}
	}

	if toolName == "web_fetch" {
		if url, ok := args["url"].(string); ok {
			// Truncate long URLs
			if len(url) > 50 {
				url = url[:47] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, url)
		}
	}

	return fmt.Sprintf("\033[1m%s:\033[0m %s", label, displayName)
}

// FormatDenyResult returns the tool result message when a tool is denied.
func FormatDenyResult(toolName string, reason string) string {
	if reason != "" {
		return fmt.Sprintf("User denied execution of %s. Reason: %s", toolName, reason)
	}
	return fmt.Sprintf("User denied execution of %s.", toolName)
}
