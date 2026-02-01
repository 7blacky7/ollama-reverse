// cmd_auth.go - Authentifizierungs-Commands: signin, signout
// Hauptfunktionen: SigninHandler, SignoutHandler
package cmd

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

// SigninHandler - Meldet den Benutzer bei ollama.com an
func SigninHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	user, err := client.Whoami(cmd.Context())
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
			fmt.Println("You need to be signed in to Ollama to run Cloud models.")
			fmt.Println()

			if aErr.SigninURL != "" {
				fmt.Printf(ConnectInstructions, aErr.SigninURL)
			}
			return nil
		}
		return err
	}

	if user != nil && user.Name != "" {
		fmt.Printf("You are already signed in as user '%s'\n", user.Name)
		fmt.Println()
		return nil
	}

	return nil
}

// SignoutHandler - Meldet den Benutzer von ollama.com ab
func SignoutHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	err = client.Signout(cmd.Context())
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
			fmt.Println("You are not signed in to ollama.com")
			fmt.Println()
			return nil
		} else {
			return err
		}
	}

	fmt.Println("You have signed out of ollama.com")
	fmt.Println()
	return nil
}

// newSigninCmd - Erstellt den signin Command
func newSigninCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "signin",
		Short:   "Sign in to ollama.com",
		Args:    cobra.ExactArgs(0),
		PreRunE: checkServerHeartbeat,
		RunE:    SigninHandler,
	}
}

// newSignoutCmd - Erstellt den signout Command
func newSignoutCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "signout",
		Short:   "Sign out from ollama.com",
		Args:    cobra.ExactArgs(0),
		PreRunE: checkServerHeartbeat,
		RunE:    SignoutHandler,
	}
}
