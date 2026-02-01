// MODUL: options
// ZWECK: Functional Options Pattern fuer Vision Encoder Konfiguration
// INPUT: Optionale Konfigurationsparameter (Device, Threads, BatchSize, Quantization)
// OUTPUT: LoadOptions Struct mit Konfiguration
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: runtime (Standard-Library)
// HINWEISE: Verwendet Functional Options Pattern fuer erweiterbare Konfiguration

package vision

import (
	"errors"
	"runtime"
)

// ============================================================================
// LoadOptions - Zentrale Konfigurationsstruktur
// ============================================================================

// LoadOptions enthaelt die Konfiguration fuer das Laden eines Vision Encoders.
type LoadOptions struct {
	Device       string // Compute-Backend: "cpu", "cuda", "metal"
	Threads      int    // Anzahl CPU-Threads
	BatchSize    int    // Batch-Groesse fuer Encoding
	Quantization string // Quantisierung: "f16", "q8_0", "q4_k_m"
	GPULayers    int    // Anzahl GPU-Layers (-1 fuer alle)
	MainGPU      int    // Index des Haupt-GPUs
	UseMmap      bool   // Memory-Mapping aktivieren
	UseMlock     bool   // Memory-Locking aktivieren
}

// Option ist eine funktionale Option fuer LoadOptions.
type Option func(*LoadOptions)

// ============================================================================
// Fehler-Definitionen fuer Options
// ============================================================================

var (
	ErrInvalidDevice       = errors.New("vision: invalid device")
	ErrInvalidQuantization = errors.New("vision: invalid quantization")
	ErrInvalidThreads      = errors.New("vision: invalid thread count")
	ErrInvalidBatchSize    = errors.New("vision: invalid batch size")
)

// ============================================================================
// DefaultLoadOptions - Standard-Konfiguration
// ============================================================================

// DefaultLoadOptions gibt eine Standard-Konfiguration zurueck.
// - Device: "cpu" (sicherster Default)
// - Threads: Anzahl CPU-Kerne
// - BatchSize: 1 (einzelne Bilder)
// - Quantization: "f16" (beste Qualitaet)
func DefaultLoadOptions() LoadOptions {
	return LoadOptions{
		Device:       DeviceCPU,
		Threads:      runtime.NumCPU(),
		BatchSize:    1,
		Quantization: QuantF16,
		GPULayers:    -1, // alle Layer auf GPU wenn verfuegbar
		MainGPU:      0,
		UseMmap:      true,
		UseMlock:     false,
	}
}

// ============================================================================
// Functional Options - Builder-Funktionen
// ============================================================================

// WithDevice setzt das Compute-Backend.
// Gueltige Werte: "cpu", "cuda", "metal"
func WithDevice(device string) Option {
	return func(o *LoadOptions) {
		o.Device = device
	}
}

// WithThreads setzt die Anzahl der CPU-Threads.
// Werte <= 0 werden durch runtime.NumCPU() ersetzt.
func WithThreads(n int) Option {
	return func(o *LoadOptions) {
		if n > 0 {
			o.Threads = n
		}
	}
}

// WithBatchSize setzt die Batch-Groesse fuer Encoding.
// Werte <= 0 werden ignoriert (Default: 1).
func WithBatchSize(n int) Option {
	return func(o *LoadOptions) {
		if n > 0 {
			o.BatchSize = n
		}
	}
}

// WithQuantization setzt die Quantisierungsstufe.
// Gueltige Werte: "f16", "q8_0", "q4_k_m"
func WithQuantization(q string) Option {
	return func(o *LoadOptions) {
		o.Quantization = q
	}
}

// WithGPULayers setzt die Anzahl der GPU-Layers.
// -1 bedeutet alle Layer auf GPU.
func WithGPULayers(n int) Option {
	return func(o *LoadOptions) {
		o.GPULayers = n
	}
}

// WithMainGPU setzt den Index des Haupt-GPUs.
func WithMainGPU(gpu int) Option {
	return func(o *LoadOptions) {
		if gpu >= 0 {
			o.MainGPU = gpu
		}
	}
}

// WithMmap aktiviert/deaktiviert Memory-Mapping.
func WithMmap(enabled bool) Option {
	return func(o *LoadOptions) {
		o.UseMmap = enabled
	}
}

// WithMlock aktiviert/deaktiviert Memory-Locking.
func WithMlock(enabled bool) Option {
	return func(o *LoadOptions) {
		o.UseMlock = enabled
	}
}

// ============================================================================
// Apply - Options auf LoadOptions anwenden
// ============================================================================

// Apply wendet alle Options auf LoadOptions an.
func (o *LoadOptions) Apply(opts ...Option) {
	for _, opt := range opts {
		opt(o)
	}
}

// ============================================================================
// Konstanten fuer Device und Quantization
// ============================================================================

const (
	// Device-Konstanten
	DeviceCPU   = "cpu"
	DeviceCUDA  = "cuda"
	DeviceMetal = "metal"

	// Quantization-Konstanten
	QuantF16  = "f16"
	QuantQ8_0 = "q8_0"
	QuantQ4KM = "q4_k_m"
)

// ============================================================================
// Validation - Konfiguration validieren
// ============================================================================

// Validate prueft ob die LoadOptions gueltig sind.
func (o *LoadOptions) Validate() error {
	// Device pruefen
	switch o.Device {
	case DeviceCPU, DeviceCUDA, DeviceMetal:
		// gueltig
	default:
		return ErrInvalidDevice
	}

	// Quantization pruefen
	switch o.Quantization {
	case QuantF16, QuantQ8_0, QuantQ4KM:
		// gueltig
	default:
		return ErrInvalidQuantization
	}

	// Threads pruefen
	if o.Threads <= 0 {
		return ErrInvalidThreads
	}

	// BatchSize pruefen
	if o.BatchSize <= 0 {
		return ErrInvalidBatchSize
	}

	return nil
}
