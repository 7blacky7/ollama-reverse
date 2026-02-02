// MODUL: vision-bench
// ZWECK: Standalone CLI Tool fuer Vision Encoder Benchmarks
// INPUT: CLI-Argumente (--backend, --model, --batch-sizes, etc.)
// OUTPUT: Benchmark-Report (Console, JSON, Markdown)
// NEBENEFFEKTE: Modell-Laden, GPU-Zugriff, Dateisystem-Schreibzugriff
// ABHAENGIGKEITEN: vision/benchmark Package
// HINWEISE: Aufruf: vision-bench --backend cuda --model siglip --batch-sizes 1,8,32

package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/vision/benchmark"
)

// ============================================================================
// Version und Hilfe
// ============================================================================

const (
	version   = "1.0.0"
	toolName  = "vision-bench"
)

// usage gibt die Hilfe-Nachricht aus.
func usage() {
	fmt.Printf(`%s v%s - Vision Encoder Benchmark Tool

VERWENDUNG:
    %s [OPTIONS]

OPTIONEN:
    --backend <backends>    Backends kommagetrennt: cpu,cuda,metal
                            Standard: cpu

    --model <modelle>       Modelle kommagetrennt: siglip,clip,dinov2,nomic,openclip
                            Standard: siglip

    --batch-sizes <sizes>   Batch-Groessen kommagetrennt: 1,4,8,16,32
                            Standard: 1,4,8,16,32

    --model-path <pfad>     Pfad zur GGUF-Modelldatei
                            Ohne Angabe: Simulierte Ergebnisse

    --iterations <n>        Anzahl Benchmark-Iterationen
                            Standard: 100

    --warmup <n>            Anzahl Warmup-Laeufe
                            Standard: 10

    --output-json <pfad>    Pfad fuer JSON-Report
    --output-md <pfad>      Pfad fuer Markdown-Report

    --list-backends         Verfuegbare Backends auflisten
    --verbose               Detaillierte Ausgabe
    --help                  Diese Hilfe anzeigen
    --version               Version anzeigen

BEISPIELE:
    # CPU Benchmark mit Standard-Einstellungen
    %s

    # GPU Benchmark mit CUDA
    %s --backend cuda --model siglip,clip

    # Vollstaendiger Benchmark mit allen Backends
    %s --backend cpu,cuda,metal --batch-sizes 1,8,32

    # Benchmark mit Modell-Datei und Report-Export
    %s --model-path model.gguf --output-json report.json --output-md report.md

    # Verfuegbare Backends anzeigen
    %s --list-backends

BACKENDS:
    cpu     - CPU-basierte Inferenz (immer verfuegbar)
    cuda    - NVIDIA GPU (erfordert CUDA)
    metal   - Apple GPU (nur macOS)

MODELLE:
    siglip   - SigLIP Vision Encoder (Google)
    clip     - CLIP Vision Encoder (OpenAI)
    dinov2   - DINOv2 Vision Encoder (Meta)
    nomic    - Nomic Vision Encoder
    openclip - OpenCLIP Vision Encoder

AUSGABE:
    Der Benchmark gibt folgende Metriken aus:
    - Durchschnittliche Latenz pro Bild
    - P95 Latenz (95. Perzentil)
    - Durchsatz (Bilder pro Sekunde)
    - Speicherverbrauch

    Bei mehreren Backends werden Vergleichstabellen erstellt.

`, toolName, version, toolName, toolName, toolName, toolName, toolName, toolName)
}

// ============================================================================
// Main Funktion
// ============================================================================

func main() {
	// Hilfe und Version pruefen
	if containsFlag(os.Args, "--help", "-h") {
		usage()
		os.Exit(0)
	}

	if containsFlag(os.Args, "--version", "-v") {
		fmt.Printf("%s v%s\n", toolName, version)
		os.Exit(0)
	}

	// Banner ausgeben
	printBanner()

	// CLI-Argumente parsen und Benchmark ausfuehren
	if err := runBenchmark(); err != nil {
		fmt.Fprintf(os.Stderr, "FEHLER: %v\n", err)
		os.Exit(1)
	}
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// containsFlag prueft ob ein Flag in den Argumenten enthalten ist.
func containsFlag(args []string, flags ...string) bool {
	for _, arg := range args {
		for _, flag := range flags {
			if arg == flag {
				return true
			}
		}
	}
	return false
}

// printBanner gibt das Tool-Banner aus.
func printBanner() {
	fmt.Println()
	fmt.Println("  _   _(_)___(_) ___  _ __        | |__   ___ _ __   ___| |__  ")
	fmt.Println(" \\ \\ / / / __| |/ _ \\| '_ \\ _____| '_ \\ / _ \\ '_ \\ / __| '_ \\ ")
	fmt.Println("  \\ V /| \\__ \\ | (_) | | | |_____| |_) |  __/ | | | (__| | | |")
	fmt.Println("   \\_/ |_|___/_|\\___/|_| |_|     |_.__/ \\___|_| |_|\\___|_| |_|")
	fmt.Printf("                                                    v%s\n", version)
	fmt.Println()
}

// runBenchmark fuehrt den Benchmark aus.
func runBenchmark() error {
	// Konfiguration aus CLI parsen
	config, err := benchmark.ParseFlags(os.Args[1:])
	if err != nil {
		return fmt.Errorf("argumente parsen: %w", err)
	}

	// Runner erstellen und ausfuehren
	runner := benchmark.NewRunner(config)
	return runner.Run()
}

// ============================================================================
// Zusaetzliche Kommandos
// ============================================================================

// printQuickStats gibt eine schnelle Uebersicht aus (ohne vollstaendigen Benchmark).
func printQuickStats() {
	fmt.Println("Schnellstatistik:")
	fmt.Println("-----------------")
	fmt.Println("Diese Funktion erfordert ein geladenes Modell.")
	fmt.Println("Verwenden Sie --model-path fuer vollstaendige Benchmarks.")
}

// validateEnvironment prueft die Laufzeitumgebung.
func validateEnvironment() error {
	// Pruefe ob mindestens ein Backend verfuegbar ist
	// CPU ist immer verfuegbar, daher gibt diese Pruefung immer true zurueck
	return nil
}
