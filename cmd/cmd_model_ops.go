// cmd_model_ops.go - Model-Operationen: Copy und Delete Handler
// Hauptfunktionen: CopyHandler, DeleteHandler, newCopyCmd, newDeleteCmd
package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

// CopyHandler - Kopiert ein Modell
func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(cmd.Context(), &req); err != nil {
		return err
	}
	fmt.Printf("copied '%s' to '%s'\n", args[0], args[1])
	return nil
}

// DeleteHandler - Loescht ein oder mehrere Modelle
func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	for _, arg := range args {
		if err := loadOrUnloadModel(cmd, &runOptions{
			Model:     arg,
			KeepAlive: &api.Duration{Duration: 0},
		}); err != nil {
			if !strings.Contains(strings.ToLower(err.Error()), "not found") {
				fmt.Fprintf(os.Stderr, "Warning: unable to stop model '%s'\n", arg)
			}
		}

		if err := client.Delete(cmd.Context(), &api.DeleteRequest{Name: arg}); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", arg)
	}
	return nil
}

// newCopyCmd - Erstellt den copy Command
func newCopyCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "cp SOURCE DESTINATION",
		Short:   "Copy a model",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    CopyHandler,
	}
}

// newDeleteCmd - Erstellt den delete Command
func newDeleteCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "rm MODEL [MODEL...]",
		Short:   "Remove a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    DeleteHandler,
	}
}
