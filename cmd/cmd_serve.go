// cmd_serve.go - Server und Keypair Funktionen
// Hauptfunktionen: RunServer, initializeKeypair, versionHandler
package cmd

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/runner"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/version"
)

// RunServer - Startet den Ollama-Server
func RunServer(_ *cobra.Command, _ []string) error {
	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", envconfig.Host().Host)
	if err != nil {
		return err
	}

	err = server.Serve(ln)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return err
}

// initializeKeypair - Erstellt SSH-Schluessel wenn nicht vorhanden
func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
		cryptoPublicKey, cryptoPrivateKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privateKeyBytes, err := ssh.MarshalPrivateKey(cryptoPrivateKey, "")
		if err != nil {
			return err
		}

		if err := os.MkdirAll(filepath.Dir(privKeyPath), 0o755); err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		if err := os.WriteFile(privKeyPath, pem.EncodeToMemory(privateKeyBytes), 0o600); err != nil {
			return err
		}

		sshPublicKey, err := ssh.NewPublicKey(cryptoPublicKey)
		if err != nil {
			return err
		}

		publicKeyBytes := ssh.MarshalAuthorizedKey(sshPublicKey)

		if err := os.WriteFile(pubKeyPath, publicKeyBytes, 0o644); err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", publicKeyBytes)
	}
	return nil
}

// versionHandler - Zeigt die Version an
func versionHandler(cmd *cobra.Command, _ []string) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	serverVersion, err := client.Version(cmd.Context())
	if err != nil {
		fmt.Println("Warning: could not connect to a running Ollama instance")
	}

	if serverVersion != "" {
		fmt.Printf("ollama version is %s\n", serverVersion)
	}

	if serverVersion != version.Version {
		fmt.Printf("Warning: client version is %s\n", version.Version)
	}
}

// newServeCmd - Erstellt den serve Command
func newServeCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start Ollama",
		Args:    cobra.ExactArgs(0),
		RunE:    RunServer,
	}
}

// newRunnerCmd - Erstellt den versteckten runner Command
func newRunnerCmd() *cobra.Command {
	runnerCmd := &cobra.Command{
		Use:    "runner",
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runner.Execute(os.Args[1:])
		},
		FParseErrWhitelist: cobra.FParseErrWhitelist{UnknownFlags: true},
	}
	runnerCmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		_ = runner.Execute(args[1:])
	})
	return runnerCmd
}
