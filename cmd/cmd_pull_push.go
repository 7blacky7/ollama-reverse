// cmd_pull_push.go - Pull und Push Commands
// Hauptfunktionen: PullHandler, PushHandler
package cmd

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
)

// PullHandler - Laedt ein Modell von einer Registry herunter
func PullHandler(cmd *cobra.Command, args []string) error {
	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if resp.Completed == 0 {
				return nil
			}

			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				name, isDigest := strings.CutPrefix(resp.Digest, "sha256:")
				name = strings.TrimSpace(name)
				if isDigest {
					name = name[:min(12, len(name))]
				}
				bar = progress.NewBar(fmt.Sprintf("pulling %s:", name), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: args[0], Insecure: insecure}
	return client.Pull(cmd.Context(), &request, fn)
}

// PushHandler - Laedt ein Modell in eine Registry hoch
func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	n := model.ParseName(args[0])
	if strings.HasSuffix(n.Host, ".ollama.ai") || strings.HasSuffix(n.Host, ".ollama.com") {
		_, err := client.Whoami(cmd.Context())
		if err != nil {
			var aErr api.AuthorizationError
			if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
				fmt.Println("You need to be signed in to push models to ollama.com.")
				fmt.Println()

				if aErr.SigninURL != "" {
					fmt.Printf(ConnectInstructions, aErr.SigninURL)
				}
				return nil
			}

			return err
		}
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("pushing %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PushRequest{Name: args[0], Insecure: insecure}

	if err := client.Push(cmd.Context(), &request, fn); err != nil {
		if spinner != nil {
			spinner.Stop()
		}
		errStr := strings.ToLower(err.Error())
		if strings.Contains(errStr, "access denied") || strings.Contains(errStr, "unauthorized") {
			return errors.New("you are not authorized to push to this namespace, create the model under a namespace you own")
		}
		return err
	}

	p.Stop()
	spinner.Stop()

	destination := n.String()
	if strings.HasSuffix(n.Host, ".ollama.ai") || strings.HasSuffix(n.Host, ".ollama.com") {
		destination = "https://ollama.com/" + strings.TrimSuffix(n.DisplayShortest(), ":latest")
	}
	fmt.Printf("\nYou can find your model at:\n\n")
	fmt.Printf("\t%s\n", destination)

	return nil
}

// newPullCmd - Erstellt den pull Command
func newPullCmd() *cobra.Command {
	pullCmd := &cobra.Command{
		Use:     "pull MODEL",
		Short:   "Pull a model from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	return pullCmd
}

// newPushCmd - Erstellt den push Command
func newPushCmd() *cobra.Command {
	pushCmd := &cobra.Command{
		Use:     "push MODEL",
		Short:   "Push a model to a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PushHandler,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	return pushCmd
}
