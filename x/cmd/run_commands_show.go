// run_commands_show.go - /show Kommando-Handler
//
// Dieses Modul enthält:
// - handleShowCommand: /show Modell-Info-Kommando (info, license, modelfile, etc.)

package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

// handleShowCommand verarbeitet das /show Kommando.
func handleShowCommand(cmd *cobra.Command, args []string, modelName string, options map[string]any, system string) {
	if len(args) > 1 {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			fmt.Println("error: couldn't connect to ollama server")
			return
		}
		req := &api.ShowRequest{
			Name:    modelName,
			Options: options,
		}
		resp, err := client.Show(cmd.Context(), req)
		if err != nil {
			fmt.Println("error: couldn't get model")
			return
		}

		switch args[1] {
		case "info":
			showModelInfo(modelName, resp)
		case "license":
			showLicense(resp)
		case "modelfile":
			fmt.Println(resp.Modelfile)
		case "parameters":
			showParameters(resp, options)
		case "system":
			showSystemMessage(resp, system)
		case "template":
			showTemplate(resp)
		default:
			fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
		}
	} else {
		fmt.Println("Usage: /show <info|license|modelfile|parameters|system|template>")
	}
}

// showModelInfo zeigt grundlegende Modell-Informationen an.
func showModelInfo(modelName string, resp *api.ShowResponse) {
	fmt.Fprintf(os.Stderr, "  Model\n")
	fmt.Fprintf(os.Stderr, "    %-16s %s\n", "Name", modelName)
	if resp.Details.Family != "" {
		fmt.Fprintf(os.Stderr, "    %-16s %s\n", "Family", resp.Details.Family)
	}
	if resp.Details.ParameterSize != "" {
		fmt.Fprintf(os.Stderr, "    %-16s %s\n", "Parameter Size", resp.Details.ParameterSize)
	}
	if resp.Details.QuantizationLevel != "" {
		fmt.Fprintf(os.Stderr, "    %-16s %s\n", "Quantization", resp.Details.QuantizationLevel)
	}
	if len(resp.Capabilities) > 0 {
		caps := make([]string, len(resp.Capabilities))
		for i, c := range resp.Capabilities {
			caps[i] = string(c)
		}
		fmt.Fprintf(os.Stderr, "    %-16s %s\n", "Capabilities", strings.Join(caps, ", "))
	}
	fmt.Fprintln(os.Stderr)
}

// showLicense zeigt die Modell-Lizenz an.
func showLicense(resp *api.ShowResponse) {
	if resp.License == "" {
		fmt.Println("No license was specified for this model.")
	} else {
		fmt.Println(resp.License)
	}
}

// showParameters zeigt Modell- und User-Parameter an.
func showParameters(resp *api.ShowResponse, options map[string]any) {
	fmt.Println("Model defined parameters:")
	if resp.Parameters == "" {
		fmt.Println("  No additional parameters were specified.")
	} else {
		for _, l := range strings.Split(resp.Parameters, "\n") {
			fmt.Printf("  %s\n", l)
		}
	}
	if len(options) > 0 {
		fmt.Println("\nUser defined parameters:")
		for k, v := range options {
			fmt.Printf("  %-30s %v\n", k, v)
		}
	}
}

// showSystemMessage zeigt die System-Nachricht an.
func showSystemMessage(resp *api.ShowResponse, system string) {
	switch {
	case system != "":
		fmt.Println(system + "\n")
	case resp.System != "":
		fmt.Println(resp.System + "\n")
	default:
		fmt.Println("No system message was specified for this model.")
	}
}

// showTemplate zeigt das Prompt-Template an.
func showTemplate(resp *api.ShowResponse) {
	if resp.Template != "" {
		fmt.Println(resp.Template)
	} else {
		fmt.Println("No prompt template was specified for this model.")
	}
}
