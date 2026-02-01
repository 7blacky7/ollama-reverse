// MODUL: vision-benchmark/main
// ZWECK: CLI-Tool fuer Vision Encoder Benchmarks
// INPUT: CLI-Flags (--encoder, --model, --iterations, etc.)
// OUTPUT: Benchmark-Ergebnisse (Terminal/CSV)
// NEBENEFFEKTE: Laedt Vision-Modelle, schreibt optionale CSV-Dateien
// ABHAENGIGKEITEN: vision, vision/benchmark, flag (stdlib)
// HINWEISE: Erfordert gueltige GGUF-Modelldatei

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/vision"
	"github.com/ollama/ollama/vision/benchmark"
	// Encoder-Registrierung erfolgt via init() in den jeweiligen Packages
	// Import je nach Build-Target:
	// _ "github.com/ollama/ollama/vision/clip"
	// _ "github.com/ollama/ollama/vision/evaclip"
	// _ "github.com/ollama/ollama/vision/nomic"
	// _ "github.com/ollama/ollama/vision/openclip"
)

// ============================================================================
// CLI-Konfiguration
// ============================================================================

// cliOptions enthaelt alle CLI-Flags.
type cliOptions struct {
	encoder    string
	modelPath  string
	iterations int
	warmup     int
	batchSizes string
	imageSizes string
	format     string
	output     string
	device     string
	threads    int
	verbose    bool
}

// ============================================================================
// Main-Funktion
// ============================================================================

func main() {
	opts := parseFlags()

	if err := validateOptions(opts); err != nil {
		fmt.Fprintf(os.Stderr, "Fehler: %v\n", err)
		flag.Usage()
		os.Exit(1)
	}

	if err := runBenchmark(opts); err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark fehlgeschlagen: %v\n", err)
		os.Exit(1)
	}
}

// ============================================================================
// Flag-Parsing
// ============================================================================

// parseFlags parst die CLI-Argumente und gibt cliOptions zurueck.
func parseFlags() cliOptions {
	opts := cliOptions{}

	flag.StringVar(&opts.encoder, "encoder", "", "Encoder-Typ (siglip, clip, nomic, evaclip)")
	flag.StringVar(&opts.modelPath, "model", "", "Pfad zur GGUF-Modelldatei")
	flag.IntVar(&opts.iterations, "iterations", 100, "Anzahl Benchmark-Iterationen")
	flag.IntVar(&opts.warmup, "warmup", 10, "Anzahl Warmup-Laeufe")
	flag.StringVar(&opts.batchSizes, "batch", "1,4,8,16", "Batch-Groessen (kommasepariert)")
	flag.StringVar(&opts.imageSizes, "sizes", "224x224,384x384", "Bildgroessen (kommasepariert)")
	flag.StringVar(&opts.format, "format", "table", "Ausgabeformat: table, markdown, csv")
	flag.StringVar(&opts.output, "output", "", "CSV-Ausgabedatei (optional)")
	flag.StringVar(&opts.device, "device", "cpu", "Compute-Backend: cpu, cuda, metal")
	flag.IntVar(&opts.threads, "threads", 0, "CPU-Threads (0 = auto)")
	flag.BoolVar(&opts.verbose, "v", false, "Ausfuehrliche Ausgabe")

	flag.Usage = printUsage
	flag.Parse()

	return opts
}

// printUsage gibt die Hilfe-Nachricht aus.
func printUsage() {
	fmt.Fprintf(os.Stderr, "Vision Encoder Benchmark Tool\n\n")
	fmt.Fprintf(os.Stderr, "Verwendung: vision-benchmark [OPTIONEN]\n\n")
	fmt.Fprintf(os.Stderr, "Optionen:\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nBeispiele:\n")
	fmt.Fprintf(os.Stderr, "  vision-benchmark --encoder siglip --model model.gguf --iterations 50\n")
	fmt.Fprintf(os.Stderr, "  vision-benchmark --encoder clip --model clip.gguf --format csv --output results.csv\n")
}

// ============================================================================
// Validierung
// ============================================================================

// validateOptions prueft die CLI-Optionen auf Gueltigkeit.
func validateOptions(opts cliOptions) error {
	if opts.encoder == "" {
		return fmt.Errorf("--encoder ist erforderlich")
	}
	if opts.modelPath == "" {
		return fmt.Errorf("--model ist erforderlich")
	}
	if _, err := os.Stat(opts.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("modelldatei nicht gefunden: %s", opts.modelPath)
	}
	return nil
}

// ============================================================================
// Benchmark-Ausfuehrung
// ============================================================================

// runBenchmark fuehrt den eigentlichen Benchmark aus.
func runBenchmark(opts cliOptions) error {
	// Encoder laden
	encoder, err := loadEncoder(opts)
	if err != nil {
		return fmt.Errorf("encoder laden: %w", err)
	}
	defer encoder.Close()

	if opts.verbose {
		info := encoder.ModelInfo()
		fmt.Printf("Encoder geladen: %s (Dim: %d)\n", info.Name, info.EmbeddingDim)
	}

	// Benchmark-Config erstellen
	config := buildConfig(opts)

	// Benchmark ausfuehren
	results := benchmark.RunBenchmark(encoder, config)

	// Ergebnisse ausgeben
	return outputResults(results, opts)
}

// loadEncoder laedt den Vision-Encoder basierend auf CLI-Optionen.
func loadEncoder(opts cliOptions) (vision.VisionEncoder, error) {
	loadOpts := []vision.Option{
		vision.WithDevice(opts.device),
	}
	if opts.threads > 0 {
		loadOpts = append(loadOpts, vision.WithThreads(opts.threads))
	}
	return vision.NewEncoder(opts.encoder, opts.modelPath, loadOpts...)
}

// buildConfig erstellt die Benchmark-Konfiguration aus CLI-Optionen.
func buildConfig(opts cliOptions) benchmark.BenchmarkConfig {
	return benchmark.BenchmarkConfig{
		Iterations: opts.iterations,
		WarmupRuns: opts.warmup,
		BatchSizes: parseIntList(opts.batchSizes),
		ImageSizes: parseStringList(opts.imageSizes),
		Verbose:    opts.verbose,
	}
}

// outputResults gibt die Ergebnisse im gewuenschten Format aus.
func outputResults(results []benchmark.BenchmarkResult, opts cliOptions) error {
	switch opts.format {
	case "csv":
		if opts.output != "" {
			return benchmark.ExportCSV(results, opts.output)
		}
		return benchmark.WriteCSV(os.Stdout, results)
	case "markdown":
		benchmark.PrintMarkdown(os.Stdout, results)
	default:
		benchmark.PrintResults(results)
	}

	// Zusaetzlich CSV exportieren wenn --output angegeben
	if opts.output != "" && opts.format != "csv" {
		return benchmark.ExportCSV(results, opts.output)
	}
	return nil
}

// ============================================================================
// Parsing-Hilfsfunktionen
// ============================================================================

// parseIntList parst eine kommaseparierte Liste von Integers.
func parseIntList(s string) []int {
	parts := strings.Split(s, ",")
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%d", &n); err == nil && n > 0 {
			result = append(result, n)
		}
	}
	return result
}

// parseStringList parst eine kommaseparierte Liste von Strings.
func parseStringList(s string) []string {
	parts := strings.Split(s, ",")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}
