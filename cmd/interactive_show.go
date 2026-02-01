// interactive_show.go - /show Befehl Handler
// Verarbeitet alle /show Unterbefehle fuer Model-Informationen
package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

// handleShowCommand verarbeitet den /show Befehl
func handleShowCommand(cmd *cobra.Command, opts *runOptions, line string) error {
	args := strings.Fields(line)
	if len(args) <= 1 {
		usageShow()
		return nil
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Println("error: couldn't connect to ollama server")
		return err
	}
	req := &api.ShowRequest{
		Name:    opts.Model,
		System:  opts.System,
		Options: opts.Options,
	}
	resp, err := client.Show(cmd.Context(), req)
	if err != nil {
		fmt.Println("error: couldn't get model")
		return err
	}

	switch args[1] {
	case "info":
		_ = showInfo(resp, false, os.Stderr)
	case "license":
		if resp.License == "" {
			fmt.Println("No license was specified for this model.")
		} else {
			fmt.Println(resp.License)
		}
	case "modelfile":
		fmt.Println(resp.Modelfile)
	case "parameters":
		showParameters(resp, opts)
	case "system":
		showSystem(resp, opts)
	case "template":
		if resp.Template != "" {
			fmt.Println(resp.Template)
		} else {
			fmt.Println("No prompt template was specified for this model.")
		}
	default:
		fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
	}
	return nil
}

// showParameters zeigt die Parameter-Informationen an
func showParameters(resp *api.ShowResponse, opts *runOptions) {
	fmt.Println("Model defined parameters:")
	if resp.Parameters == "" {
		fmt.Println("  No additional parameters were specified for this model.")
	} else {
		for _, l := range strings.Split(resp.Parameters, "\n") {
			fmt.Printf("  %s\n", l)
		}
	}
	fmt.Println()
	if len(opts.Options) > 0 {
		fmt.Println("User defined parameters:")
		for k, v := range opts.Options {
			fmt.Printf("  %-*s %v\n", 30, k, v)
		}
		fmt.Println()
	}
}

// showSystem zeigt die System-Nachricht an
func showSystem(resp *api.ShowResponse, opts *runOptions) {
	switch {
	case opts.System != "":
		fmt.Println(opts.System + "\n")
	case resp.System != "":
		fmt.Println(resp.System + "\n")
	default:
		fmt.Println("No system message was specified for this model.")
	}
}
