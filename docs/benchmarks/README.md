# Vision Encoder Benchmark Suite

GPU-Backend Benchmark-Suite fuer Vision Encoder. Vergleicht CPU, CUDA und Metal Performance fuer verschiedene Vision Encoder Modelle.

## Installation

```bash
# Tool bauen
go build -o vision-bench ./tools/vision-bench

# Mit CUDA-Support
go build -tags cuda -o vision-bench ./tools/vision-bench

# Mit Metal-Support (macOS)
go build -tags metal -o vision-bench ./tools/vision-bench
```

## Verwendung

### Grundlegende Benchmarks

```bash
# CPU Benchmark (Standard)
./vision-bench

# GPU Benchmark mit CUDA
./vision-bench --backend cuda

# Alle Backends vergleichen
./vision-bench --backend cpu,cuda,metal
```

### Modell-Auswahl

```bash
# Einzelnes Modell
./vision-bench --model siglip

# Mehrere Modelle
./vision-bench --model siglip,clip,dinov2

# Mit Modell-Datei
./vision-bench --model siglip --model-path /path/to/model.gguf
```

### Batch-Groessen

```bash
# Standard: 1,4,8,16,32
./vision-bench --batch-sizes 1,4,8,16,32

# Nur grosse Batches
./vision-bench --batch-sizes 16,32,64
```

### Report-Export

```bash
# JSON-Report
./vision-bench --output-json benchmark_results.json

# Markdown-Report
./vision-bench --output-md benchmark_results.md

# Beide Formate
./vision-bench --output-json results.json --output-md results.md
```

### Verfuegbare Backends anzeigen

```bash
./vision-bench --list-backends
```

## CLI-Optionen

| Option | Beschreibung | Standard |
|--------|--------------|----------|
| `--backend` | Backends (cpu,cuda,metal) | cpu |
| `--model` | Modelle (siglip,clip,dinov2) | siglip |
| `--batch-sizes` | Batch-Groessen | 1,4,8,16,32 |
| `--model-path` | Pfad zur GGUF-Datei | - |
| `--iterations` | Benchmark-Iterationen | 100 |
| `--warmup` | Warmup-Laeufe | 10 |
| `--output-json` | JSON-Report-Pfad | - |
| `--output-md` | Markdown-Report-Pfad | - |
| `--verbose` | Detaillierte Ausgabe | false |
| `--list-backends` | Backends auflisten | false |

## Unterstuetzte Backends

### CPU
- Immer verfuegbar
- Multi-Threading mit GGML
- Optimal fuer kleine Batch-Groessen

### CUDA (NVIDIA GPU)
- Erfordert NVIDIA GPU mit CUDA-Support
- Build-Tag: `cuda`
- Compute Capability >= 5.0 empfohlen

### Metal (Apple GPU)
- Nur macOS mit Apple Silicon oder AMD GPU
- Build-Tag: `metal`
- Unified Memory-Optimierungen

## Unterstuetzte Modelle

| Modell | Beschreibung | Embedding-Dim |
|--------|--------------|---------------|
| siglip | SigLIP Vision Encoder (Google) | 768/1024 |
| clip | CLIP Vision Encoder (OpenAI) | 512/768 |
| dinov2 | DINOv2 Vision Encoder (Meta) | 768/1024 |
| nomic | Nomic Vision Encoder | 768 |
| openclip | OpenCLIP Vision Encoder | 512/768 |

## Beispiel-Ergebnisse

### CPU vs CUDA vs Metal (RTX 4090, M2 Max)

```
Vision Encoder Benchmark Ergebnisse
====================================
Encoder         Backend    Size     Batch  Avg Latenz   P95 Latenz   Throughput Memory
-------------------------------------------------------------------------------
siglip          cpu        224x224  1      12.34ms      15.21ms      81.1       128.5 MB
siglip          cuda       224x224  1      1.23ms       1.45ms       813.0      256.0 MB
siglip          metal      224x224  1      1.89ms       2.12ms       529.1      192.0 MB
siglip          cpu        224x224  32     198.45ms     215.32ms     161.2      512.0 MB
siglip          cuda       224x224  32     8.92ms       10.15ms      3587.4     1024.0 MB
siglip          metal      224x224  32     15.67ms      18.23ms      2042.1     768.0 MB
```

### Backend-Vergleich

| Encoder | Size | Batch | CPU | CUDA | Metal | Winner | Speedup |
|---------|------|-------|-----|------|-------|--------|---------|
| siglip | 224x224 | 1 | 81.1 | 813.0 | 529.1 | cuda | 10.0x |
| siglip | 224x224 | 32 | 161.2 | 3587.4 | 2042.1 | cuda | 22.3x |
| clip | 224x224 | 1 | 75.3 | 756.2 | 489.5 | cuda | 10.0x |
| dinov2 | 224x224 | 1 | 68.9 | 692.1 | 445.8 | cuda | 10.0x |

## Metriken

### Latenz
- **Avg Latenz**: Durchschnittliche Zeit pro Bild
- **Min/Max Latenz**: Beste/schlechteste Zeit
- **P95 Latenz**: 95. Perzentil (wichtig fuer Worst-Case)

### Durchsatz
- **Throughput**: Bilder pro Sekunde
- Berechnung: `(BatchSize * Iterationen) / Gesamtzeit`

### Speicher
- **Memory Used**: Speicherverbrauch waehrend Benchmark
- Bei GPU: Device Memory
- Bei CPU: Heap-Allokation

## Programmatische Nutzung

```go
package main

import (
    "github.com/ollama/ollama/vision"
    "github.com/ollama/ollama/vision/benchmark"
)

func main() {
    // Encoder laden
    encoder, _ := vision.NewEncoder(
        "siglip",
        "/path/to/model.gguf",
        vision.WithDevice("cuda"),
    )
    defer encoder.Close()

    // Benchmark konfigurieren
    config := benchmark.BenchmarkConfig{
        Iterations: 100,
        WarmupRuns: 10,
        BatchSizes: []int{1, 8, 32},
        ImageSizes: []string{"224x224"},
    }

    // Benchmark ausfuehren
    results := benchmark.RunBenchmark(encoder, config)

    // Ergebnisse ausgeben
    benchmark.PrintResults(results)

    // Report erstellen
    report := benchmark.NewReport(results, config, sysInfo)
    report.ExportJSON("results.json")
    report.ExportMarkdown("results.md")
}
```

## Tipps fuer genaue Messungen

1. **Warmup-Phase**: Mindestens 10 Warmup-Laeufe fuer stabile Messungen
2. **Iterationen**: 100+ Iterationen fuer statistische Signifikanz
3. **Keine Last**: Andere GPU-Prozesse beenden
4. **Konsistente Umgebung**: Gleiche Hardware/Software fuer Vergleiche
5. **Kuehlung**: GPU-Throttling durch Ueberhitzung vermeiden

## Fehlerbehebung

### CUDA nicht erkannt
```bash
# CUDA-Installation pruefen
nvidia-smi

# Build mit korrekten Tags
go build -tags cuda ./tools/vision-bench
```

### Metal nicht erkannt
```bash
# Nur auf macOS verfuegbar
# Apple Silicon oder AMD GPU erforderlich
go build -tags metal ./tools/vision-bench
```

### Out of Memory
- Batch-Groesse reduzieren
- GPU-Layers anpassen mit `--gpu-layers`

## Lizenz

Teil des Ollama-Projekts. Siehe LICENSE im Root-Verzeichnis.
