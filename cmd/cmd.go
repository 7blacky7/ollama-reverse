// cmd.go - Haupt-CLI Setup und Root Command
// Hauptfunktionen: NewCLI, appendEnvDocs
package cmd

import (
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/containerd/console"
	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen"
)

// ConnectInstructions - Anmeldehinweis fuer Cloud-Modelle
const ConnectInstructions = "To sign in, navigate to:\n    %s\n\n"

// appendEnvDocs - Fuegt Umgebungsvariablen-Dokumentation zum Command hinzu
func appendEnvDocs(cmd *cobra.Command, envs []envconfig.EnvVar) {
	if len(envs) == 0 {
		return
	}

	envUsage := `
Environment Variables:
`
	for _, e := range envs {
		envUsage += fmt.Sprintf("      %-24s   %s\n", e.Name, e.Description)
	}

	cmd.SetUsageTemplate(cmd.UsageTemplate() + envUsage)
}

// NewCLI - Erstellt das Haupt-CLI mit allen Commands
func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cobra.EnableCommandSorting = false

	if runtime.GOOS == "windows" && term.IsTerminal(int(os.Stdout.Fd())) {
		console.ConsoleFromFile(os.Stdin) //nolint:errcheck
	}

	rootCmd := &cobra.Command{
		Use:           "ollama",
		Short:         "Large language model runner",
		SilenceUsage:  true,
		SilenceErrors: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		Run: func(cmd *cobra.Command, args []string) {
			if version, _ := cmd.Flags().GetBool("version"); version {
				versionHandler(cmd, args)
				return
			}

			cmd.Print(cmd.UsageString())
		},
	}

	rootCmd.Flags().BoolP("version", "v", false, "Show version information")

	// Commands erstellen
	createCmd := newCreateCmd()
	showCmd := newShowCmd()
	runCmd := newRunCmd()
	stopCmd := newStopCmd()
	serveCmd := newServeCmd()
	pullCmd := newPullCmd()
	pushCmd := newPushCmd()
	signinCmd := newSigninCmd()
	signoutCmd := newSignoutCmd()
	listCmd := newListCmd()
	psCmd := newPsCmd()
	copyCmd := newCopyCmd()
	deleteCmd := newDeleteCmd()
	runnerCmd := newRunnerCmd()

	// Environment-Dokumentation hinzufuegen
	envVars := envconfig.AsMap()
	envs := []envconfig.EnvVar{envVars["OLLAMA_HOST"]}

	for _, cmd := range []*cobra.Command{
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		serveCmd,
	} {
		switch cmd {
		case runCmd:
			imagegen.AppendFlagsDocs(cmd)
			appendEnvDocs(cmd, []envconfig.EnvVar{envVars["OLLAMA_HOST"], envVars["OLLAMA_NOHISTORY"]})
		case serveCmd:
			appendEnvDocs(cmd, []envconfig.EnvVar{
				envVars["OLLAMA_DEBUG"],
				envVars["OLLAMA_HOST"],
				envVars["OLLAMA_CONTEXT_LENGTH"],
				envVars["OLLAMA_KEEP_ALIVE"],
				envVars["OLLAMA_MAX_LOADED_MODELS"],
				envVars["OLLAMA_MAX_QUEUE"],
				envVars["OLLAMA_MODELS"],
				envVars["OLLAMA_NUM_PARALLEL"],
				envVars["OLLAMA_NOPRUNE"],
				envVars["OLLAMA_ORIGINS"],
				envVars["OLLAMA_SCHED_SPREAD"],
				envVars["OLLAMA_FLASH_ATTENTION"],
				envVars["OLLAMA_KV_CACHE_TYPE"],
				envVars["OLLAMA_LLM_LIBRARY"],
				envVars["OLLAMA_GPU_OVERHEAD"],
				envVars["OLLAMA_LOAD_TIMEOUT"],
			})
		default:
			appendEnvDocs(cmd, envs)
		}
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		signinCmd,
		signoutCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		runnerCmd,
		config.LaunchCmd(checkServerHeartbeat),
	)

	return rootCmd
}
