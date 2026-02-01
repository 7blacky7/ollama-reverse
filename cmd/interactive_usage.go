// interactive_usage.go - Hilfe-Texte und Usage-Funktionen
// Zeigt Hilfe fuer Befehle, Parameter und Tastenkuerzel an
package cmd

import (
	"fmt"
	"os"
	"path/filepath"
)

// usage zeigt die allgemeine Hilfe an
func usage(opts runOptions) {
	fmt.Fprintln(os.Stderr, "Available Commands:")
	fmt.Fprintln(os.Stderr, "  /set            Set session variables")
	fmt.Fprintln(os.Stderr, "  /show           Show model information")
	fmt.Fprintln(os.Stderr, "  /load <model>   Load a session or model")
	fmt.Fprintln(os.Stderr, "  /save <model>   Save your current session")
	fmt.Fprintln(os.Stderr, "  /clear          Clear session context")
	fmt.Fprintln(os.Stderr, "  /bye            Exit")
	fmt.Fprintln(os.Stderr, "  /?, /help       Help for a command")
	fmt.Fprintln(os.Stderr, "  /? shortcuts    Help for keyboard shortcuts")

	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Use \"\"\" to begin a multi-line message.")

	if opts.MultiModal {
		fmt.Fprintf(os.Stderr, "Use %s to include .jpg, .png, or .webp images.\n", filepath.FromSlash("/path/to/file"))
	}

	fmt.Fprintln(os.Stderr, "")
}

// usageSet zeigt die Hilfe fuer /set Befehle an
func usageSet() {
	fmt.Fprintln(os.Stderr, "Available Commands:")
	fmt.Fprintln(os.Stderr, "  /set parameter ...     Set a parameter")
	fmt.Fprintln(os.Stderr, "  /set system <string>   Set system message")
	fmt.Fprintln(os.Stderr, "  /set history           Enable history")
	fmt.Fprintln(os.Stderr, "  /set nohistory         Disable history")
	fmt.Fprintln(os.Stderr, "  /set wordwrap          Enable wordwrap")
	fmt.Fprintln(os.Stderr, "  /set nowordwrap        Disable wordwrap")
	fmt.Fprintln(os.Stderr, "  /set format json       Enable JSON mode")
	fmt.Fprintln(os.Stderr, "  /set noformat          Disable formatting")
	fmt.Fprintln(os.Stderr, "  /set verbose           Show LLM stats")
	fmt.Fprintln(os.Stderr, "  /set quiet             Disable LLM stats")
	fmt.Fprintln(os.Stderr, "  /set think             Enable thinking")
	fmt.Fprintln(os.Stderr, "  /set nothink           Disable thinking")
	fmt.Fprintln(os.Stderr, "")
}

// usageShortcuts zeigt die Tastenkuerzel-Hilfe an
func usageShortcuts() {
	fmt.Fprintln(os.Stderr, "Available keyboard shortcuts:")
	fmt.Fprintln(os.Stderr, "  Ctrl + a            Move to the beginning of the line (Home)")
	fmt.Fprintln(os.Stderr, "  Ctrl + e            Move to the end of the line (End)")
	fmt.Fprintln(os.Stderr, "   Alt + b            Move back (left) one word")
	fmt.Fprintln(os.Stderr, "   Alt + f            Move forward (right) one word")
	fmt.Fprintln(os.Stderr, "  Ctrl + k            Delete the sentence after the cursor")
	fmt.Fprintln(os.Stderr, "  Ctrl + u            Delete the sentence before the cursor")
	fmt.Fprintln(os.Stderr, "  Ctrl + w            Delete the word before the cursor")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "  Ctrl + l            Clear the screen")
	fmt.Fprintln(os.Stderr, "  Ctrl + c            Stop the model from responding")
	fmt.Fprintln(os.Stderr, "  Ctrl + d            Exit ollama (/bye)")
	fmt.Fprintln(os.Stderr, "")
}

// usageShow zeigt die Hilfe fuer /show Befehle an
func usageShow() {
	fmt.Fprintln(os.Stderr, "Available Commands:")
	fmt.Fprintln(os.Stderr, "  /show info         Show details for this model")
	fmt.Fprintln(os.Stderr, "  /show license      Show model license")
	fmt.Fprintln(os.Stderr, "  /show modelfile    Show Modelfile for this model")
	fmt.Fprintln(os.Stderr, "  /show parameters   Show parameters for this model")
	fmt.Fprintln(os.Stderr, "  /show system       Show system message")
	fmt.Fprintln(os.Stderr, "  /show template     Show prompt template")
	fmt.Fprintln(os.Stderr, "")
}

// usageParameters zeigt die Hilfe fuer Parameter-Einstellungen an
func usageParameters() {
	fmt.Fprintln(os.Stderr, "Available Parameters:")
	fmt.Fprintln(os.Stderr, "  /set parameter seed <int>             Random number seed")
	fmt.Fprintln(os.Stderr, "  /set parameter num_predict <int>      Max number of tokens to predict")
	fmt.Fprintln(os.Stderr, "  /set parameter top_k <int>            Pick from top k num of tokens")
	fmt.Fprintln(os.Stderr, "  /set parameter top_p <float>          Pick token based on sum of probabilities")
	fmt.Fprintln(os.Stderr, "  /set parameter min_p <float>          Pick token based on top token probability * min_p")
	fmt.Fprintln(os.Stderr, "  /set parameter num_ctx <int>          Set the context size")
	fmt.Fprintln(os.Stderr, "  /set parameter temperature <float>    Set creativity level")
	fmt.Fprintln(os.Stderr, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
	fmt.Fprintln(os.Stderr, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
	fmt.Fprintln(os.Stderr, "  /set parameter num_gpu <int>          The number of layers to send to the GPU")
	fmt.Fprintln(os.Stderr, "  /set parameter stop <string> <string> ...   Set the stop parameters")
	fmt.Fprintln(os.Stderr, "")
}
