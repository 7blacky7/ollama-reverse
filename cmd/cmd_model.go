// cmd_model.go - Model-Verwaltung: create, copy, delete
// Hauptfunktionen: CreateHandler, CopyHandler, DeleteHandler
package cmd

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/types/syncmap"
	"github.com/ollama/ollama/x/create"
	xcreateclient "github.com/ollama/ollama/x/create/client"
)

var errModelfileNotFound = errors.New("specified Modelfile wasn't found")

// getModelfileName - Findet den Modelfile-Pfad
func getModelfileName(cmd *cobra.Command) (string, error) {
	filename, _ := cmd.Flags().GetString("file")

	if filename == "" {
		filename = "Modelfile"
	}

	absName, err := filepath.Abs(filename)
	if err != nil {
		return "", err
	}

	_, err = os.Stat(absName)
	if err != nil {
		return "", err
	}

	return absName, nil
}

// CreateHandler - Erstellt ein neues Modell
func CreateHandler(cmd *cobra.Command, args []string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	modelName := args[0]
	name := model.ParseName(modelName)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	experimental, _ := cmd.Flags().GetBool("experimental")
	if experimental {
		return handleExperimentalCreate(cmd, modelName, p)
	}

	return handleStandardCreate(cmd, modelName, p)
}

// handleExperimentalCreate - Erstellt Modell im experimentellen Modus
func handleExperimentalCreate(cmd *cobra.Command, modelName string, p *progress.Progress) error {
	var reader io.Reader
	filename, err := getModelfileName(cmd)
	if os.IsNotExist(err) || filename == "" {
		reader = strings.NewReader("FROM .\n")
	} else if err != nil {
		return err
	} else {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer f.Close()
		reader = f
	}

	modelfile, err := parser.ParseFile(reader)
	if err != nil {
		return fmt.Errorf("failed to parse Modelfile: %w", err)
	}

	var modelDir string
	mfConfig := &xcreateclient.ModelfileConfig{}

	for _, cmd := range modelfile.Commands {
		switch cmd.Name {
		case "model":
			modelDir = cmd.Args
		case "template":
			mfConfig.Template = cmd.Args
		case "system":
			mfConfig.System = cmd.Args
		case "license":
			mfConfig.License = cmd.Args
		}
	}

	if modelDir == "" {
		modelDir = "."
	}

	if !filepath.IsAbs(modelDir) && filename != "" {
		modelDir = filepath.Join(filepath.Dir(filename), modelDir)
	}

	quantize, _ := cmd.Flags().GetString("quantize")
	return xcreateclient.CreateModel(xcreateclient.CreateOptions{
		ModelName: modelName,
		ModelDir:  modelDir,
		Quantize:  quantize,
		Modelfile: mfConfig,
	}, p)
}

// handleStandardCreate - Erstellt Modell im Standard-Modus
func handleStandardCreate(cmd *cobra.Command, modelName string, p *progress.Progress) error {
	var reader io.Reader

	filename, err := getModelfileName(cmd)
	if os.IsNotExist(err) {
		if filename == "" {
			if create.IsTensorModelDir(".") {
				quantize, _ := cmd.Flags().GetString("quantize")
				return xcreateclient.CreateModel(xcreateclient.CreateOptions{
					ModelName: modelName,
					ModelDir:  ".",
					Quantize:  quantize,
				}, p)
			}
			reader = strings.NewReader("FROM .\n")
		} else {
			return errModelfileNotFound
		}
	} else if err != nil {
		return err
	} else {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		reader = f
		defer f.Close()
	}

	modelfile, err := parser.ParseFile(reader)
	if err != nil {
		return err
	}

	status := "gathering model components"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	req, err := modelfile.CreateRequest(filepath.Dir(filename))
	if err != nil {
		return err
	}
	spinner.Stop()

	req.Model = modelName
	quantize, _ := cmd.Flags().GetString("quantize")
	if quantize != "" {
		req.Quantize = quantize
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	var g errgroup.Group
	g.SetLimit(max(runtime.GOMAXPROCS(0)-1, 1))

	files := syncmap.NewSyncMap[string, string]()
	for f, digest := range req.Files {
		g.Go(func() error {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}
			files.Store(filepath.Base(f), digest)
			return nil
		})
	}

	adapters := syncmap.NewSyncMap[string, string]()
	for f, digest := range req.Adapters {
		g.Go(func() error {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}
			adapters.Store(filepath.Base(f), digest)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	req.Files = files.Items()
	req.Adapters = adapters.Items()

	bars := make(map[string]*progress.Bar)
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("pulling %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()
			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	if err := client.Create(cmd.Context(), req, fn); err != nil {
		if strings.Contains(err.Error(), "path or Modelfile are required") {
			return fmt.Errorf("the ollama server must be updated to use `ollama create` with this client")
		}
		return err
	}

	return nil
}

// createBlob - Laedt eine Datei als Blob hoch
func createBlob(cmd *cobra.Command, client *api.Client, path string, digest string, p *progress.Progress) (string, error) {
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}

	bin, err := os.Open(realPath)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	fileInfo, err := bin.Stat()
	if err != nil {
		return "", err
	}
	fileSize := fileInfo.Size()

	var pw progressWriter
	status := fmt.Sprintf("copying file %s 0%%", digest)
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)
	defer spinner.Stop()

	done := make(chan struct{})
	defer close(done)

	go func() {
		ticker := time.NewTicker(60 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				spinner.SetMessage(fmt.Sprintf("copying file %s %d%%", digest, int(100*pw.n.Load()/fileSize)))
			case <-done:
				spinner.SetMessage(fmt.Sprintf("copying file %s 100%%", digest))
				return
			}
		}
	}()

	if err := client.CreateBlob(cmd.Context(), digest, io.TeeReader(bin, &pw)); err != nil {
		return "", err
	}
	return digest, nil
}

type progressWriter struct {
	n atomic.Int64
}

func (w *progressWriter) Write(p []byte) (n int, err error) {
	w.n.Add(int64(len(p)))
	return len(p), nil
}

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

// newCreateCmd - Erstellt den create Command
func newCreateCmd() *cobra.Command {
	createCmd := &cobra.Command{
		Use:   "create MODEL",
		Short: "Create a model",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(cmd *cobra.Command, args []string) error {
			if experimental, _ := cmd.Flags().GetBool("experimental"); experimental {
				return nil
			}
			return checkServerHeartbeat(cmd, args)
		},
		RunE: CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "", "Name of the Modelfile (default \"Modelfile\")")
	createCmd.Flags().StringP("quantize", "q", "", "Quantize model to this level (e.g. q4_K_M)")
	createCmd.Flags().Bool("experimental", false, "Enable experimental safetensors model creation")

	return createCmd
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
