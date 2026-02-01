// Package siglip - Functional Options fuer Model-Konfiguration.
//
// Diese Datei enthaelt:
// - options Struct: Interne Konfigurationsstruktur
// - defaultOptions: Standard-Konfiguration
// - WithBackend, WithLogLevel, WithEmbedFormat, etc.: Functional Options
package siglip

import "runtime"

// ============================================================================
// Options Struct
// ============================================================================

// Option ist eine funktionale Option fuer LoadModel.
type Option func(*options)

// options speichert die Modell-Konfiguration.
type options struct {
	backend     Backend
	logLevel    LogLevel
	embedFormat EmbedFormat
	nThreads    int
	nGPULayers  int
	mainGPU     int
	useMmap     bool
	useMlock    bool
	batchSize   int
}

// defaultOptions gibt die Standard-Konfiguration zurueck.
func defaultOptions() *options {
	return &options{
		backend:     BackendCPU,
		logLevel:    LogInfo,
		embedFormat: EmbedF32,
		nThreads:    runtime.NumCPU(),
		nGPULayers:  -1, // alle Layer auf GPU
		mainGPU:     0,
		useMmap:     true,
		useMlock:    false,
		batchSize:   1,
	}
}

// ============================================================================
// Functional Options
// ============================================================================

// WithBackend setzt das Compute-Backend.
func WithBackend(backend Backend) Option {
	return func(o *options) {
		o.backend = backend
	}
}

// WithLogLevel setzt das Log-Level.
func WithLogLevel(level LogLevel) Option {
	return func(o *options) {
		o.logLevel = level
	}
}

// WithEmbedFormat setzt das Embedding-Format.
func WithEmbedFormat(format EmbedFormat) Option {
	return func(o *options) {
		o.embedFormat = format
	}
}

// WithThreads setzt die Anzahl der CPU-Threads.
func WithThreads(n int) Option {
	return func(o *options) {
		o.nThreads = n
	}
}

// WithGPULayers setzt die Anzahl der GPU-Layers (-1 fuer alle).
func WithGPULayers(n int) Option {
	return func(o *options) {
		o.nGPULayers = n
	}
}

// WithMainGPU setzt den Haupt-GPU Index.
func WithMainGPU(gpu int) Option {
	return func(o *options) {
		o.mainGPU = gpu
	}
}

// WithMmap aktiviert/deaktiviert Memory-Mapping.
func WithMmap(enabled bool) Option {
	return func(o *options) {
		o.useMmap = enabled
	}
}

// WithMlock aktiviert/deaktiviert Memory-Locking.
func WithMlock(enabled bool) Option {
	return func(o *options) {
		o.useMlock = enabled
	}
}

// WithBatchSize setzt die Batch-Groesse.
func WithBatchSize(size int) Option {
	return func(o *options) {
		o.batchSize = size
	}
}
