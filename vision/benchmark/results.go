// MODUL: results
// ZWECK: Formatierung und Export von Benchmark-Ergebnissen
// INPUT: BenchmarkResult Slices
// OUTPUT: Formatierte Ausgabe (Terminal, CSV, Markdown)
// NEBENEFFEKTE: Dateisystem-Schreibzugriff bei ExportCSV
// ABHAENGIGKEITEN: fmt, os, encoding/csv (stdlib)
// HINWEISE: CSV-Export verwendet Semikolon als Trennzeichen fuer DE-Kompatibilitaet

package benchmark

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

// ============================================================================
// Terminal-Ausgabe
// ============================================================================

// PrintResults gibt Benchmark-Ergebnisse formatiert auf stdout aus.
func PrintResults(results []BenchmarkResult) {
	PrintResultsTo(os.Stdout, results)
}

// PrintResultsTo gibt Ergebnisse auf einen beliebigen Writer aus.
func PrintResultsTo(w io.Writer, results []BenchmarkResult) {
	if len(results) == 0 {
		fmt.Fprintln(w, "Keine Ergebnisse vorhanden.")
		return
	}

	printTableHeader(w)

	for _, r := range results {
		printResultRow(w, r)
	}
}

// printTableHeader gibt den Tabellenkopf aus.
func printTableHeader(w io.Writer) {
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Vision Encoder Benchmark Ergebnisse")
	fmt.Fprintln(w, "====================================")
	fmt.Fprintf(w, "%-15s %-10s %-8s %-6s %-12s %-12s %-12s %-10s\n",
		"Encoder", "Backend", "Size", "Batch", "Avg Latenz", "P95 Latenz", "Throughput", "Memory")
	fmt.Fprintln(w, "-------------------------------------------------------------------------------")
}

// printResultRow gibt eine Ergebniszeile aus.
func printResultRow(w io.Writer, r BenchmarkResult) {
	fmt.Fprintf(w, "%-15s %-10s %-8s %-6d %-12s %-12s %-10.1f %-10s\n",
		truncateString(r.EncoderName, 15),
		r.Backend,
		r.ImageSize,
		r.BatchSize,
		formatDuration(r.AvgLatency),
		formatDuration(r.P95Latency),
		r.Throughput,
		formatBytes(r.MemoryUsed),
	)
}

// ============================================================================
// Markdown-Ausgabe
// ============================================================================

// PrintMarkdown gibt Ergebnisse als Markdown-Tabelle aus.
func PrintMarkdown(w io.Writer, results []BenchmarkResult) {
	if len(results) == 0 {
		fmt.Fprintln(w, "_Keine Ergebnisse vorhanden._")
		return
	}

	printMarkdownHeader(w)

	for _, r := range results {
		printMarkdownRow(w, r)
	}
}

// printMarkdownHeader gibt den Markdown-Tabellenkopf aus.
func printMarkdownHeader(w io.Writer) {
	fmt.Fprintln(w, "| Encoder | Backend | Size | Batch | Avg Latenz | P95 | Throughput | Memory |")
	fmt.Fprintln(w, "|---------|---------|------|-------|------------|-----|------------|--------|")
}

// printMarkdownRow gibt eine Markdown-Tabellenzeile aus.
func printMarkdownRow(w io.Writer, r BenchmarkResult) {
	fmt.Fprintf(w, "| %s | %s | %s | %d | %s | %s | %.1f img/s | %s |\n",
		r.EncoderName,
		r.Backend,
		r.ImageSize,
		r.BatchSize,
		formatDuration(r.AvgLatency),
		formatDuration(r.P95Latency),
		r.Throughput,
		formatBytes(r.MemoryUsed),
	)
}

// ============================================================================
// CSV-Export
// ============================================================================

// ExportCSV exportiert Ergebnisse als CSV-Datei.
func ExportCSV(results []BenchmarkResult, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("csv-datei erstellen: %w", err)
	}
	defer f.Close()

	return WriteCSV(f, results)
}

// WriteCSV schreibt Ergebnisse als CSV auf einen Writer.
func WriteCSV(w io.Writer, results []BenchmarkResult) error {
	cw := csv.NewWriter(w)
	cw.Comma = ';' // Semikolon fuer DE-Excel-Kompatibilitaet

	// Header schreiben
	header := []string{
		"encoder", "backend", "image_size", "batch_size", "iterations",
		"avg_latency_ms", "min_latency_ms", "max_latency_ms", "p95_latency_ms",
		"throughput_img_s", "memory_bytes", "embedding_dim",
	}
	if err := cw.Write(header); err != nil {
		return err
	}

	// Daten schreiben
	for _, r := range results {
		row := buildCSVRow(r)
		if err := cw.Write(row); err != nil {
			return err
		}
	}

	cw.Flush()
	return cw.Error()
}

// buildCSVRow erstellt eine CSV-Zeile aus einem BenchmarkResult.
func buildCSVRow(r BenchmarkResult) []string {
	return []string{
		r.EncoderName,
		r.Backend,
		r.ImageSize,
		strconv.Itoa(r.BatchSize),
		strconv.Itoa(r.Iterations),
		strconv.FormatFloat(float64(r.AvgLatency.Microseconds())/1000, 'f', 3, 64),
		strconv.FormatFloat(float64(r.MinLatency.Microseconds())/1000, 'f', 3, 64),
		strconv.FormatFloat(float64(r.MaxLatency.Microseconds())/1000, 'f', 3, 64),
		strconv.FormatFloat(float64(r.P95Latency.Microseconds())/1000, 'f', 3, 64),
		strconv.FormatFloat(r.Throughput, 'f', 2, 64),
		strconv.FormatUint(r.MemoryUsed, 10),
		strconv.Itoa(r.EmbeddingDim),
	}
}

// ============================================================================
// Formatierungs-Hilfsfunktionen
// ============================================================================

// formatDuration formatiert eine Duration fuer menschliche Lesbarkeit.
func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fus", float64(d.Nanoseconds())/1000)
	}
	if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d.Microseconds())/1000)
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}

// formatBytes formatiert Bytes fuer menschliche Lesbarkeit.
func formatBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// truncateString kuerzt einen String auf maxLen Zeichen.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
