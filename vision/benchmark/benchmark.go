// MODUL: benchmark
// ZWECK: Benchmark-Suite fuer Vision Encoder mit Latenz-, Durchsatz- und Speichermessung
// INPUT: VisionEncoder, BenchmarkConfig
// OUTPUT: BenchmarkResult mit detaillierten Metriken
// NEBENEFFEKTE: CPU/GPU-Last waehrend Benchmark, Speicherallokation
// ABHAENGIGKEITEN: vision (VisionEncoder), runtime (Speichermessung)
// HINWEISE: Warmup-Laeufe sind wichtig fuer stabile Messungen

package benchmark

import (
	"fmt"
	"runtime"
	"sort"
	"time"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Datenstrukturen - Ergebnisse
// ============================================================================

// BenchmarkResult enthaelt das Ergebnis eines einzelnen Benchmark-Laufs.
type BenchmarkResult struct {
	EncoderName  string        // Name des getesteten Encoders
	Backend      string        // Compute-Backend (cpu, cuda, metal)
	ImageSize    string        // Bildgroesse z.B. "224x224"
	BatchSize    int           // Batch-Groesse
	Iterations   int           // Anzahl Durchlaeufe
	TotalTime    time.Duration // Gesamtzeit aller Iterationen
	AvgLatency   time.Duration // Durchschnittliche Latenz pro Bild
	MinLatency   time.Duration // Minimale Latenz
	MaxLatency   time.Duration // Maximale Latenz
	P95Latency   time.Duration // 95. Perzentil Latenz
	Throughput   float64       // Bilder pro Sekunde
	MemoryUsed   uint64        // Speicherverbrauch in Bytes
	EmbeddingDim int           // Embedding-Dimension
}

// ============================================================================
// Datenstrukturen - Konfiguration
// ============================================================================

// BenchmarkConfig definiert die Parameter fuer einen Benchmark-Lauf.
type BenchmarkConfig struct {
	Iterations int      // Anzahl Messungen (ohne Warmup)
	WarmupRuns int      // Anzahl Warmup-Laeufe (nicht gemessen)
	BatchSizes []int    // Zu testende Batch-Groessen
	ImageSizes []string // Zu testende Bildgroessen z.B. "224x224"
	Verbose    bool     // Detaillierte Ausgabe waehrend Benchmark
}

// DefaultConfig gibt eine Standard-Benchmark-Konfiguration zurueck.
func DefaultConfig() BenchmarkConfig {
	return BenchmarkConfig{
		Iterations: 100,
		WarmupRuns: 10,
		BatchSizes: []int{1, 4, 8, 16, 32},
		ImageSizes: []string{"224x224", "384x384", "512x512"},
		Verbose:    false,
	}
}

// ============================================================================
// Haupt-Benchmark-Funktionen
// ============================================================================

// RunBenchmark fuehrt einen vollstaendigen Benchmark fuer einen Encoder aus.
// Testet alle Kombinationen aus BatchSizes und ImageSizes.
func RunBenchmark(encoder vision.VisionEncoder, config BenchmarkConfig) []BenchmarkResult {
	var results []BenchmarkResult
	info := encoder.ModelInfo()

	for _, imageSize := range config.ImageSizes {
		width, height := parseImageSize(imageSize)
		if width == 0 || height == 0 {
			continue
		}

		for _, batchSize := range config.BatchSizes {
			result := benchmarkSingleConfig(encoder, info, width, height, batchSize, config)
			results = append(results, result)
		}
	}

	return results
}

// RunAllBenchmarks benchmarkt alle registrierten Encoder.
// Gibt eine Map von Encoder-Name zu Ergebnis-Liste zurueck.
func RunAllBenchmarks(config BenchmarkConfig) map[string][]BenchmarkResult {
	results := make(map[string][]BenchmarkResult)
	encoderNames := vision.ListFromDefault()

	for _, name := range encoderNames {
		// Hinweis: Encoder muss extern geladen werden, da wir keinen Modell-Pfad haben
		// Diese Funktion ist fuer Integration mit Modell-Loader gedacht
		results[name] = nil
	}

	return results
}

// ============================================================================
// Interne Benchmark-Logik
// ============================================================================

// benchmarkSingleConfig fuehrt Benchmark fuer eine Konfiguration aus.
func benchmarkSingleConfig(encoder vision.VisionEncoder, info vision.ModelInfo, width, height, batchSize int, config BenchmarkConfig) BenchmarkResult {
	imageSize := fmt.Sprintf("%dx%d", width, height)

	// Testdaten generieren
	testBatch := GenerateTestBatch(width, height, batchSize)

	// Warmup-Phase
	runWarmup(encoder, testBatch, config.WarmupRuns)

	// GC erzwingen vor Messung
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Benchmark-Phase mit Latenzmessung
	latencies := measureLatencies(encoder, testBatch, config.Iterations)

	// Speicher nach Messung
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	return buildResult(info, imageSize, batchSize, latencies, memBefore, memAfter, config)
}

// runWarmup fuehrt Warmup-Iterationen aus (ohne Messung).
func runWarmup(encoder vision.VisionEncoder, batch [][]byte, runs int) {
	for i := 0; i < runs; i++ {
		_, _ = encoder.EncodeBatch(batch)
	}
}

// measureLatencies misst die Latenzen fuer jede Iteration.
func measureLatencies(encoder vision.VisionEncoder, batch [][]byte, iterations int) []time.Duration {
	latencies := make([]time.Duration, 0, iterations)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, _ = encoder.EncodeBatch(batch)
		latencies = append(latencies, time.Since(start))
	}

	return latencies
}

// buildResult erstellt das BenchmarkResult aus den Messungen.
func buildResult(info vision.ModelInfo, imageSize string, batchSize int, latencies []time.Duration, memBefore, memAfter runtime.MemStats, config BenchmarkConfig) BenchmarkResult {
	stats := calculateStats(latencies)
	totalImages := batchSize * config.Iterations

	return BenchmarkResult{
		EncoderName:  info.Name,
		Backend:      "cpu", // TODO: Aus LoadOptions extrahieren
		ImageSize:    imageSize,
		BatchSize:    batchSize,
		Iterations:   config.Iterations,
		TotalTime:    stats.total,
		AvgLatency:   stats.avg / time.Duration(batchSize),
		MinLatency:   stats.min / time.Duration(batchSize),
		MaxLatency:   stats.max / time.Duration(batchSize),
		P95Latency:   stats.p95 / time.Duration(batchSize),
		Throughput:   float64(totalImages) / stats.total.Seconds(),
		MemoryUsed:   memAfter.Alloc - memBefore.Alloc,
		EmbeddingDim: info.EmbeddingDim,
	}
}

// ============================================================================
// Statistik-Hilfsfunktionen
// ============================================================================

// latencyStats enthaelt berechnete Latenz-Statistiken.
type latencyStats struct {
	total time.Duration
	avg   time.Duration
	min   time.Duration
	max   time.Duration
	p95   time.Duration
}

// calculateStats berechnet Statistiken aus Latenz-Messungen.
func calculateStats(latencies []time.Duration) latencyStats {
	if len(latencies) == 0 {
		return latencyStats{}
	}

	// Sortieren fuer Perzentile
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	// Summe berechnen
	var total time.Duration
	for _, d := range latencies {
		total += d
	}

	// P95 Index
	p95Idx := int(float64(len(sorted)) * 0.95)
	if p95Idx >= len(sorted) {
		p95Idx = len(sorted) - 1
	}

	return latencyStats{
		total: total,
		avg:   total / time.Duration(len(latencies)),
		min:   sorted[0],
		max:   sorted[len(sorted)-1],
		p95:   sorted[p95Idx],
	}
}

// ============================================================================
// Hilfsfunktionen - Parsing
// ============================================================================

// parseImageSize parsed einen String wie "224x224" zu width, height.
func parseImageSize(size string) (int, int) {
	var width, height int
	_, err := fmt.Sscanf(size, "%dx%d", &width, &height)
	if err != nil {
		return 0, 0
	}
	return width, height
}
