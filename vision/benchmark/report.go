// MODUL: report
// ZWECK: Report-Generierung fuer Benchmark-Ergebnisse (JSON, Markdown, Console)
// INPUT: BenchmarkResult Slices, ReportConfig
// OUTPUT: Formatierte Reports mit Backend-Vergleichstabellen
// NEBENEFFEKTE: Dateisystem-Schreibzugriff bei Export-Funktionen
// ABHAENGIGKEITEN: encoding/json, fmt, io, os, time (stdlib)
// HINWEISE: Erzeugt Vergleichstabellen CPU vs CUDA vs Metal

package benchmark

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"time"
)

// Report enthaelt alle Benchmark-Ergebnisse mit Metadaten.
type Report struct {
	Timestamp   time.Time         `json:"timestamp"`
	SystemInfo  SystemInfo        `json:"system_info"`
	Config      BenchmarkConfig   `json:"config"`
	Results     []BenchmarkResult `json:"results"`
	Comparisons []BackendCompare  `json:"comparisons"`
	Summary     ReportSummary     `json:"summary"`
}

// SystemInfo enthaelt Systeminformationen zum Benchmark.
type SystemInfo struct {
	OS            string   `json:"os"`
	Arch          string   `json:"arch"`
	CPUCores      int      `json:"cpu_cores"`
	AvailableGPUs []string `json:"available_gpus"`
}

// ReportSummary fasst die wichtigsten Ergebnisse zusammen.
type ReportSummary struct {
	FastestBackend string  `json:"fastest_backend"`
	BestThroughput float64 `json:"best_throughput"`
	TotalTests     int     `json:"total_tests"`
	PassedTests    int     `json:"passed_tests"`
}

// BackendCompare vergleicht Ergebnisse zwischen Backends.
type BackendCompare struct {
	EncoderName   string              `json:"encoder_name"`
	ImageSize     string              `json:"image_size"`
	BatchSize     int                 `json:"batch_size"`
	BackendStats  map[string]CompStat `json:"backend_stats"`
	Winner        string              `json:"winner"`
	SpeedupFactor float64             `json:"speedup_factor"`
}

// CompStat enthaelt Vergleichsstatistiken fuer ein Backend.
type CompStat struct {
	AvgLatencyMs float64 `json:"avg_latency_ms"`
	Throughput   float64 `json:"throughput"`
	MemoryMB     float64 `json:"memory_mb"`
}

// NewReport erstellt einen neuen Report aus Benchmark-Ergebnissen.
func NewReport(results []BenchmarkResult, config BenchmarkConfig, sysInfo SystemInfo) *Report {
	report := &Report{
		Timestamp:  time.Now(),
		SystemInfo: sysInfo,
		Config:     config,
		Results:    results,
	}
	report.Comparisons = generateComparisons(results)
	report.Summary = generateSummary(results, report.Comparisons)
	return report
}

// generateComparisons erstellt Backend-Vergleiche aus Ergebnissen.
func generateComparisons(results []BenchmarkResult) []BackendCompare {
	groups := make(map[string][]BenchmarkResult)
	for _, r := range results {
		key := fmt.Sprintf("%s_%s_%d", r.EncoderName, r.ImageSize, r.BatchSize)
		groups[key] = append(groups[key], r)
	}

	var comparisons []BackendCompare
	for _, group := range groups {
		if len(group) >= 2 {
			comparisons = append(comparisons, createComparison(group))
		}
	}
	return comparisons
}

// createComparison erstellt einen Vergleich aus einer Gruppe von Ergebnissen.
func createComparison(group []BenchmarkResult) BackendCompare {
	first := group[0]
	cmp := BackendCompare{
		EncoderName:  first.EncoderName,
		ImageSize:    first.ImageSize,
		BatchSize:    first.BatchSize,
		BackendStats: make(map[string]CompStat),
	}

	var bestThroughput float64
	for _, r := range group {
		cmp.BackendStats[r.Backend] = CompStat{
			AvgLatencyMs: float64(r.AvgLatency.Microseconds()) / 1000.0,
			Throughput:   r.Throughput,
			MemoryMB:     float64(r.MemoryUsed) / (1024.0 * 1024.0),
		}
		if r.Throughput > bestThroughput {
			bestThroughput = r.Throughput
			cmp.Winner = r.Backend
		}
	}

	if cpuStat, hasCPU := cmp.BackendStats["cpu"]; hasCPU && cpuStat.Throughput > 0 {
		cmp.SpeedupFactor = bestThroughput / cpuStat.Throughput
	}
	return cmp
}

// generateSummary erstellt eine Zusammenfassung.
func generateSummary(results []BenchmarkResult, comparisons []BackendCompare) ReportSummary {
	summary := ReportSummary{TotalTests: len(results), PassedTests: len(results)}

	winnerCounts := make(map[string]int)
	for _, cmp := range comparisons {
		winnerCounts[cmp.Winner]++
	}
	var maxCount int
	for backend, count := range winnerCounts {
		if count > maxCount {
			maxCount = count
			summary.FastestBackend = backend
		}
	}
	for _, r := range results {
		if r.Throughput > summary.BestThroughput {
			summary.BestThroughput = r.Throughput
		}
	}
	return summary
}

// ExportJSON exportiert den Report als JSON-Datei.
func (r *Report) ExportJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("json-datei erstellen: %w", err)
	}
	defer f.Close()
	return r.WriteJSON(f)
}

// WriteJSON schreibt den Report als JSON auf einen Writer.
func (r *Report) WriteJSON(w io.Writer) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(r)
}

// ExportMarkdown exportiert den Report als Markdown-Datei.
func (r *Report) ExportMarkdown(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("markdown-datei erstellen: %w", err)
	}
	defer f.Close()
	return r.WriteMarkdown(f)
}

// WriteMarkdown schreibt den Report als Markdown.
func (r *Report) WriteMarkdown(w io.Writer) error {
	fmt.Fprintf(w, "# Vision Encoder Benchmark Report\n\n")
	fmt.Fprintf(w, "**Datum:** %s\n\n", r.Timestamp.Format("2006-01-02 15:04:05"))

	fmt.Fprintf(w, "## Systeminfo\n\n")
	fmt.Fprintf(w, "- **OS:** %s\n- **Architektur:** %s\n- **CPU-Kerne:** %d\n",
		r.SystemInfo.OS, r.SystemInfo.Arch, r.SystemInfo.CPUCores)
	if len(r.SystemInfo.AvailableGPUs) > 0 {
		fmt.Fprintf(w, "- **GPUs:** %v\n", r.SystemInfo.AvailableGPUs)
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "## Ergebnisse\n\n")
	PrintMarkdown(w, r.Results)
	fmt.Fprintln(w)

	if len(r.Comparisons) > 0 {
		fmt.Fprintf(w, "## Backend-Vergleich\n\n")
		r.writeComparisonTable(w)
	}

	fmt.Fprintf(w, "## Zusammenfassung\n\n")
	fmt.Fprintf(w, "- **Schnellstes Backend:** %s\n", r.Summary.FastestBackend)
	fmt.Fprintf(w, "- **Beste Durchsatzrate:** %.1f img/s\n", r.Summary.BestThroughput)
	fmt.Fprintf(w, "- **Tests durchgefuehrt:** %d\n", r.Summary.TotalTests)
	return nil
}

func (r *Report) writeComparisonTable(w io.Writer) {
	fmt.Fprintln(w, "| Encoder | Size | Batch | CPU | CUDA | Metal | Winner | Speedup |")
	fmt.Fprintln(w, "|---------|------|-------|-----|------|-------|--------|---------|")
	for _, cmp := range r.Comparisons {
		fmt.Fprintf(w, "| %s | %s | %d | %s | %s | %s | %s | %.1fx |\n",
			cmp.EncoderName, cmp.ImageSize, cmp.BatchSize,
			formatThroughputCell(cmp.BackendStats["cpu"]),
			formatThroughputCell(cmp.BackendStats["cuda"]),
			formatThroughputCell(cmp.BackendStats["metal"]),
			cmp.Winner, cmp.SpeedupFactor)
	}
	fmt.Fprintln(w)
}

func formatThroughputCell(stat CompStat) string {
	if stat.Throughput == 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f", stat.Throughput)
}

// PrintConsole gibt den Report auf der Konsole aus.
func (r *Report) PrintConsole() { r.WriteConsole(os.Stdout) }

// WriteConsole schreibt den Report fuer Konsolen-Ausgabe.
func (r *Report) WriteConsole(w io.Writer) {
	fmt.Fprintln(w, "\n========================================")
	fmt.Fprintln(w, "  VISION ENCODER BENCHMARK REPORT")
	fmt.Fprintln(w, "========================================")
	fmt.Fprintf(w, "  Datum: %s\n", r.Timestamp.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(w, "  System: %s/%s, %d CPU-Kerne\n",
		r.SystemInfo.OS, r.SystemInfo.Arch, r.SystemInfo.CPUCores)
	fmt.Fprintln(w, "========================================\n")

	PrintResultsTo(w, r.Results)

	if len(r.Comparisons) > 0 {
		fmt.Fprintln(w, "\nBACKEND-VERGLEICH:\n------------------")
		for _, cmp := range r.Comparisons {
			fmt.Fprintf(w, "  %s [%s, Batch=%d]: Winner=%s (%.1fx Speedup)\n",
				cmp.EncoderName, cmp.ImageSize, cmp.BatchSize, cmp.Winner, cmp.SpeedupFactor)
		}
	}

	fmt.Fprintln(w, "\nZUSAMMENFASSUNG:")
	fmt.Fprintf(w, "  Schnellstes Backend: %s\n", r.Summary.FastestBackend)
	fmt.Fprintf(w, "  Beste Durchsatzrate: %.1f img/s\n", r.Summary.BestThroughput)
	fmt.Fprintf(w, "  Tests: %d/%d bestanden\n\n", r.Summary.PassedTests, r.Summary.TotalTests)
}

// SortByThroughput sortiert Ergebnisse nach Durchsatz (absteigend).
func SortByThroughput(results []BenchmarkResult) {
	sort.Slice(results, func(i, j int) bool { return results[i].Throughput > results[j].Throughput })
}

// SortByLatency sortiert Ergebnisse nach Latenz (aufsteigend).
func SortByLatency(results []BenchmarkResult) {
	sort.Slice(results, func(i, j int) bool { return results[i].AvgLatency < results[j].AvgLatency })
}

// FilterByBackend filtert Ergebnisse nach Backend.
func FilterByBackend(results []BenchmarkResult, backend string) []BenchmarkResult {
	var filtered []BenchmarkResult
	for _, r := range results {
		if r.Backend == backend {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

// FilterByEncoder filtert Ergebnisse nach Encoder-Name.
func FilterByEncoder(results []BenchmarkResult, encoder string) []BenchmarkResult {
	var filtered []BenchmarkResult
	for _, r := range results {
		if r.EncoderName == encoder {
			filtered = append(filtered, r)
		}
	}
	return filtered
}
