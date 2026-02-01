// runner.go - Haupteinstiegspunkt fuer den Ollama Runner
//
// Enthaelt:
// - Execute: Startet den Runner-Server

package ollamarunner

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"

	_ "github.com/ollama/ollama/model/models"
)

// Execute startet den Ollama Runner Server
func Execute(args []string) error {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	port := fs.Int("port", defaultPort, "Port to expose the server on")
	_ = fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting ollama engine")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := &Server{
		modelPath: *mpath,
		status:    llm.ServerStatusLaunched,
	}

	server.cond = sync.NewCond(&server.mu)
	server.ready.Add(1)

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /info", server.info)
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("POST /embedding", server.embeddings)
	mux.HandleFunc("POST /completion", server.completion)
	mux.HandleFunc("GET /health", server.health)

	httpServer := http.Server{
		Handler: mux,
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
		return err
	}

	return nil
}

// defaultPort ist der Standard-Port fuer den Runner
const defaultPort = 8080
