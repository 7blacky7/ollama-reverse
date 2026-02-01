// Package llamarunner - Server-Initialisierung und Modell-Laden
//
// Dieses Modul enthält:
// - loadModel: Lädt das LLM-Modell und initialisiert Kontext
// - load: HTTP-Handler für Lade-Operationen
// - Execute: Haupteinstiegspunkt für den Runner-Server
package llamarunner

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// loadModel allokiert Speicher basierend auf Parametern und lädt die Gewichte.
// Der allokierte Speicher ist Worst-Case für Text-Modelle, aber nicht für Vision.
func (s *Server) loadModel(
	params llama.ModelParams,
	mpath string,
	lpath []string,
	ppath string,
	kvSize int,
	kvCacheType string,
	flashAttention ml.FlashAttentionType,
	threads int,
	multiUserCache bool,
) {
	var err error
	s.model, err = llama.LoadModelFromFile(mpath, params)
	if err != nil {
		panic(err)
	}

	ctxParams := llama.NewContextParams(kvSize, s.batchSize, s.parallel, threads, flashAttention, kvCacheType)
	s.lc, err = llama.NewContextWithModel(s.model, ctxParams)
	if err != nil {
		panic(err)
	}

	// LoRA-Adapter anwenden
	for _, path := range lpath {
		err := s.model.ApplyLoraFromFile(s.lc, path, 1.0, threads)
		if err != nil {
			panic(err)
		}
	}

	// Vision-Projektor laden falls vorhanden
	if ppath != "" {
		var err error
		s.image, err = NewImageContext(s.lc, ppath)
		if err != nil {
			panic(err)
		}
	}

	s.cache, err = NewInputCache(s.lc, kvSize, s.parallel, multiUserCache)
	if err != nil {
		panic(err)
	}

	s.status = llm.ServerStatusReady
	s.ready.Done()
}

// load ist der Handler für Lade-Operationen vom Ollama-Server
func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	if s.status != llm.ServerStatusLaunched {
		http.Error(w, "model already loaded", http.StatusInternalServerError)
		return
	}

	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	slog.Info("load", "request", req)

	switch req.Operation {
	// LoadOperationFit und LoadOperationAlloc haben hier keine Bedeutung

	case llm.LoadOperationCommit:
		s.batchSize = req.BatchSize
		s.parallel = req.Parallel
		s.seqs = make([]*Sequence, s.parallel)
		s.seqsSem = semaphore.NewWeighted(int64(s.parallel))

		numGPU := 0
		var tensorSplit []float32
		var llamaIDs []uint64

		// GPU-Konfiguration aufbauen
		gpuIDs := llama.EnumerateGPUs()
		sort.Sort(req.GPULayers)
		for _, layers := range req.GPULayers {
			for i := range gpuIDs {
				if gpuIDs[i].DeviceID == layers.DeviceID {
					numGPU += len(layers.Layers)
					tensorSplit = append(tensorSplit, float32(len(layers.Layers)))
					llamaIDs = append(llamaIDs, gpuIDs[i].LlamaID)
				}
			}
		}

		params := llama.ModelParams{
			Devices:      llamaIDs,
			NumGpuLayers: numGPU,
			MainGpu:      req.MainGPU,
			UseMmap:      req.UseMmap && len(req.LoraPath) == 0,
			TensorSplit:  tensorSplit,
			Progress: func(progress float32) {
				s.progress = progress
			},
		}

		s.status = llm.ServerStatusLoadingModel
		go s.loadModel(params, s.modelPath, req.LoraPath, req.ProjectorPath, req.KvSize, req.KvCacheType, req.FlashAttention, req.NumThreads, req.MultiUserCache)

	case llm.LoadOperationClose:
		// No-op für uns
		if err := json.NewEncoder(w).Encode(&llm.LoadResponse{}); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		}
		return
	}

	resp := llm.LoadResponse{Success: true}
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

// Execute ist der Haupteinstiegspunkt für den LLM-Runner
func Execute(args []string) error {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	port := fs.Int("port", 8080, "Port to expose the server on")
	_ = fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting go runner")

	llama.BackendInit()

	server := &Server{
		modelPath: *mpath,
		status:    llm.ServerStatusLaunched,
	}

	server.ready.Add(1)

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
	}
	defer listener.Close()

	// HTTP-Routen registrieren
	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("/embedding", server.embeddings)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)

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
