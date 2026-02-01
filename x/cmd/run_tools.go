// run_tools.go - Tool-Ausführungslogik für Agenten-Chat
//
// Dieses Modul enthält:
// - executeToolCall: Einzelnen Tool-Call ausführen
// - handleToolApproval: Approval-Logik für Tool-Calls
// - handleWebSearchAuth: Web-Search Auth-Flow

package cmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/agent"
	"github.com/ollama/ollama/x/tools"
)

// executeToolCall führt einen einzelnen Tool-Call aus und gibt das Ergebnis zurück.
func executeToolCall(ctx context.Context, call api.ToolCall, toolRegistry *tools.Registry,
	approval *agent.ApprovalManager, opts RunOptions) api.Message {

	toolName := call.Function.Name
	args := call.Function.Arguments.ToMap()

	// Für Bash-Befehle zuerst Denylist prüfen
	if toolName == "bash" {
		if msg, blocked := checkBashDenylist(call, toolName, args); blocked {
			return msg
		}
	}

	// Approval prüfen und ggf. anfordern
	if msg, denied := handleToolApproval(toolName, args, call, approval, opts); denied {
		return msg
	}

	// Tool ausführen
	toolResult, err := toolRegistry.Execute(call)
	if err != nil {
		// Prüfen ob Web-Search Authentifizierung braucht
		if errors.Is(err, tools.ErrWebSearchAuthRequired) {
			toolResult, err = handleWebSearchAuth(ctx, call, toolRegistry)
			if err == nil {
				goto toolSuccess
			}
		}
		fmt.Fprintf(os.Stderr, "\033[1merror:\033[0m %v\n", err)
		return api.Message{
			Role:       "tool",
			Content:    fmt.Sprintf("Error: %v", err),
			ToolCallID: call.ID,
		}
	}
toolSuccess:

	// Tool-Output anzeigen (für Anzeige gekürzt)
	displayToolOutput(toolResult)

	// Output für LLM kürzen um Kontext-Überlauf zu verhindern
	toolResultForLLM := truncateToolOutput(toolResult, opts.Model)

	return api.Message{
		Role:       "tool",
		Content:    toolResultForLLM,
		ToolCallID: call.ID,
	}
}

// checkBashDenylist prüft ob ein Bash-Befehl in der Denylist ist.
// Gibt true zurück wenn blockiert, zusammen mit der Ergebnis-Message.
func checkBashDenylist(call api.ToolCall, toolName string, args map[string]any) (api.Message, bool) {
	if cmd, ok := args["command"].(string); ok {
		// Prüfen ob Befehl verweigert (gefährliches Muster)
		if denied, pattern := agent.IsDenied(cmd); denied {
			fmt.Fprintf(os.Stderr, "\033[1mblocked:\033[0m %s\n", formatToolShort(toolName, args))
			fmt.Fprintf(os.Stderr, "  matches dangerous pattern: %s\n", pattern)
			return api.Message{
				Role:       "tool",
				Content:    agent.FormatDeniedResult(cmd, pattern),
				ToolCallID: call.ID,
			}, true
		}
	}
	return api.Message{}, false
}

// handleToolApproval behandelt die Approval-Logik für Tool-Calls.
// Gibt true zurück wenn verweigert, zusammen mit der Ergebnis-Message.
func handleToolApproval(toolName string, args map[string]any, call api.ToolCall,
	approval *agent.ApprovalManager, opts RunOptions) (api.Message, bool) {

	// Im Yolo-Modus alle Approval-Prompts überspringen
	if opts.YoloMode {
		fmt.Fprintf(os.Stderr, "\033[1mrunning:\033[0m %s\n", formatToolShort(toolName, args))
		return api.Message{}, false
	}

	// Prüfen ob bereits erlaubt (verwendet Prefix-Matching für Bash-Befehle)
	if approval.IsAllowed(toolName, args) {
		// Bereits erlaubt - Running-Indikator anzeigen
		fmt.Fprintf(os.Stderr, "\033[1mrunning:\033[0m %s\n", formatToolShort(toolName, args))
		return api.Message{}, false
	}

	// Approval anfordern
	result, err := approval.RequestApproval(toolName, args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error requesting approval: %v\n", err)
		return api.Message{
			Role:       "tool",
			Content:    fmt.Sprintf("Error: %v", err),
			ToolCallID: call.ID,
		}, true
	}

	// Collapsed-Ergebnis anzeigen
	fmt.Fprintln(os.Stderr, agent.FormatApprovalResult(toolName, args, result))

	switch result.Decision {
	case agent.ApprovalDeny:
		return api.Message{
			Role:       "tool",
			Content:    agent.FormatDenyResult(toolName, result.DenyReason),
			ToolCallID: call.ID,
		}, true
	case agent.ApprovalAlways:
		approval.AddToAllowlist(toolName, args)
	}

	return api.Message{}, false
}

// handleWebSearchAuth behandelt den Auth-Flow für Web-Search.
func handleWebSearchAuth(ctx context.Context, call api.ToolCall, toolRegistry *tools.Registry) (string, error) {
	// Benutzer zum Einloggen auffordern
	fmt.Fprintf(os.Stderr, "\033[1mauth required:\033[0m web search requires authentication\n")
	result, promptErr := agent.PromptYesNo("Sign in to Ollama?")
	if promptErr == nil && result {
		// Signin-URL holen und auf Auth-Abschluss warten
		if signinErr := waitForOllamaSignin(ctx); signinErr == nil {
			// Web-Search erneut versuchen
			fmt.Fprintf(os.Stderr, "\033[90mretrying web search...\033[0m\n")
			return toolRegistry.Execute(call)
		}
	}
	return "", tools.ErrWebSearchAuthRequired
}

// displayToolOutput zeigt Tool-Output an (für Anzeige gekürzt).
func displayToolOutput(toolResult string) {
	if toolResult != "" {
		output := toolResult
		if len(output) > 300 {
			output = output[:300] + "... (truncated)"
		}
		// Ergebnis in grau, eingerückt anzeigen
		fmt.Fprintf(os.Stderr, "\033[90m  %s\033[0m\n", strings.ReplaceAll(output, "\n", "\n  "))
	}
}
