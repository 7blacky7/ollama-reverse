//go:build vision

// MODUL: config
// ZWECK: Runtime-Konfiguration fuer Vision Encoder via Environment-Variablen
// INPUT: Environment-Variablen, optional manuelle Konfiguration
// OUTPUT: VisionConfig Struct mit validierter Konfiguration
// NEBENEFFEKTE: Liest Environment-Variablen, erstellt Cache-Verzeichnis
// ABHAENGIGKEITEN: os, path/filepath, runtime, strconv, errors (Standard-Library)
// HINWEISE: Nutzt ENV-Variablen mit OLLAMA_VISION_ Prefix

package vision

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
)

// ============================================================================
// Konstanten fuer Environment-Variablen
// ============================================================================

const (
	// EnvVisionBackend definiert das Backend (onnx|gguf|auto)
	EnvVisionBackend = "OLLAMA_VISION_BACKEND"

	// EnvVisionPreferGPU aktiviert GPU-Praeferenz wenn verfuegbar
	EnvVisionPreferGPU = "OLLAMA_VISION_PREFER_GPU"

	// EnvVisionCacheDir setzt das Cache-Verzeichnis fuer Modelle
	EnvVisionCacheDir = "OLLAMA_VISION_CACHE_DIR"

	// EnvVisionModel setzt das Standard-Modell
	EnvVisionModel = "OLLAMA_VISION_MODEL"

	// EnvVisionMaxBatchSize setzt die maximale Batch-Groesse
	EnvVisionMaxBatchSize = "OLLAMA_VISION_MAX_BATCH_SIZE"

	// EnvVisionThreads setzt die Anzahl der Threads
	EnvVisionThreads = "OLLAMA_VISION_THREADS"
)

// ============================================================================
// Konstanten fuer Standardwerte
// ============================================================================

const (
	// DefaultBackendValue ist das Standard-Backend als String fuer ENV-Parsing
	DefaultBackendValue = "auto"

	// DefaultMaxBatchSize ist die Standard-Batch-Groesse
	DefaultMaxBatchSize = 4

	// DefaultCacheDirName ist der Name des Cache-Unterverzeichnisses
	DefaultCacheDirName = ".ollama/vision"

	// DefaultModel ist das Standard-Vision-Modell
	DefaultModel = "siglip"
)

// ============================================================================
// Fehler-Definitionen fuer Config
// ============================================================================

var (
	// ErrInvalidBackend wird zurueckgegeben wenn das Backend ungueltig ist
	ErrInvalidBackend = errors.New("vision/config: invalid backend, must be onnx, gguf or auto")

	// ErrInvalidMaxBatchSize wird zurueckgegeben wenn die Batch-Groesse ungueltig ist
	ErrInvalidMaxBatchSize = errors.New("vision/config: max batch size must be > 0")

	// ErrInvalidConfigThreads wird zurueckgegeben wenn die Thread-Anzahl ungueltig ist
	ErrInvalidConfigThreads = errors.New("vision/config: thread count must be > 0")

	// ErrCacheDirNotAccessible wird zurueckgegeben wenn das Cache-Verzeichnis nicht erreichbar ist
	ErrCacheDirNotAccessible = errors.New("vision/config: cache directory not accessible")
)

// ============================================================================
// VisionConfig - Zentrale Runtime-Konfiguration
// ============================================================================

// VisionConfig enthaelt die Runtime-Konfiguration fuer Vision Encoder.
type VisionConfig struct {
	// Backend definiert das Compute-Backend (onnx|gguf|auto)
	Backend string

	// PreferGPU aktiviert GPU-Beschleunigung wenn verfuegbar
	PreferGPU bool

	// CacheDir ist das Verzeichnis fuer Model-Cache
	CacheDir string

	// DefaultModel ist das Standard-Vision-Modell
	DefaultModel string

	// MaxBatchSize ist die maximale Batch-Groesse fuer Encoding
	MaxBatchSize int

	// Threads ist die Anzahl der CPU-Threads
	Threads int
}

// ============================================================================
// DefaultConfig - Standard-Konfiguration
// ============================================================================

// DefaultConfig gibt eine sinnvolle Standard-Konfiguration zurueck.
// - Backend: auto (automatische Backend-Wahl)
// - PreferGPU: true (GPU wenn verfuegbar)
// - CacheDir: ~/.ollama/vision
// - DefaultModel: siglip
// - MaxBatchSize: 4
// - Threads: Anzahl CPU-Kerne
func DefaultConfig() *VisionConfig {
	return &VisionConfig{
		Backend:      DefaultBackendValue,
		PreferGPU:    true,
		CacheDir:     GetCacheDir(),
		DefaultModel: DefaultModel,
		MaxBatchSize: DefaultMaxBatchSize,
		Threads:      runtime.NumCPU(),
	}
}

// ============================================================================
// LoadConfig - Konfiguration aus Environment laden
// ============================================================================

// LoadConfig laedt die Konfiguration aus Environment-Variablen.
// Nicht gesetzte Variablen verwenden die Standardwerte.
func LoadConfig() *VisionConfig {
	config := DefaultConfig()

	// Backend aus ENV laden
	if backend := os.Getenv(EnvVisionBackend); backend != "" {
		config.Backend = backend
	}

	// PreferGPU aus ENV laden
	if preferGPU := os.Getenv(EnvVisionPreferGPU); preferGPU != "" {
		config.PreferGPU = parseBool(preferGPU, true)
	}

	// CacheDir aus ENV laden
	if cacheDir := os.Getenv(EnvVisionCacheDir); cacheDir != "" {
		config.CacheDir = cacheDir
	}

	// DefaultModel aus ENV laden
	if model := os.Getenv(EnvVisionModel); model != "" {
		config.DefaultModel = model
	}

	// MaxBatchSize aus ENV laden
	if batchSize := os.Getenv(EnvVisionMaxBatchSize); batchSize != "" {
		if size, err := strconv.Atoi(batchSize); err == nil && size > 0 {
			config.MaxBatchSize = size
		}
	}

	// Threads aus ENV laden
	if threads := os.Getenv(EnvVisionThreads); threads != "" {
		if t, err := strconv.Atoi(threads); err == nil && t > 0 {
			config.Threads = t
		}
	}

	return config
}

// ============================================================================
// Validate - Konfiguration validieren
// ============================================================================

// Validate prueft ob die VisionConfig konsistent und gueltig ist.
// Gibt einen Fehler zurueck wenn die Konfiguration ungueltig ist.
func (c *VisionConfig) Validate() error {
	// Backend pruefen (nutzt BackendType aus selector.go)
	switch c.Backend {
	case string(BackendONNX), string(BackendGGUF), string(BackendAuto):
		// gueltig
	default:
		return ErrInvalidBackend
	}

	// MaxBatchSize pruefen
	if c.MaxBatchSize <= 0 {
		return ErrInvalidMaxBatchSize
	}

	// Threads pruefen
	if c.Threads <= 0 {
		return ErrInvalidConfigThreads
	}

	// CacheDir Zugriff pruefen (nur wenn gesetzt)
	if c.CacheDir != "" {
		if err := ensureCacheDir(c.CacheDir); err != nil {
			return ErrCacheDirNotAccessible
		}
	}

	return nil
}

// ============================================================================
// GetCacheDir - Cache-Verzeichnis mit Fallback ermitteln
// ============================================================================

// GetCacheDir gibt das Cache-Verzeichnis zurueck.
// Prueft in folgender Reihenfolge:
// 1. OLLAMA_VISION_CACHE_DIR Environment-Variable
// 2. ~/.ollama/vision (User Home Directory)
// 3. Temporaeres Verzeichnis als letzter Fallback
func GetCacheDir() string {
	// 1. Environment-Variable pruefen
	if cacheDir := os.Getenv(EnvVisionCacheDir); cacheDir != "" {
		return cacheDir
	}

	// 2. User Home Directory Fallback
	if homeDir, err := os.UserHomeDir(); err == nil {
		return filepath.Join(homeDir, DefaultCacheDirName)
	}

	// 3. Letzter Fallback: Temporaeres Verzeichnis
	return filepath.Join(os.TempDir(), "ollama-vision-cache")
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// parseBool parst einen String als Boolean mit Fallback.
// Akzeptiert: "1", "true", "yes", "on" (case-insensitive) als true
func parseBool(s string, fallback bool) bool {
	switch s {
	case "1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON":
		return true
	case "0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF":
		return false
	default:
		return fallback
	}
}

// ensureCacheDir stellt sicher dass das Cache-Verzeichnis existiert.
// Erstellt das Verzeichnis rekursiv falls noetig.
func ensureCacheDir(dir string) error {
	if dir == "" {
		return nil
	}

	// Verzeichnis erstellen falls nicht vorhanden
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Schreibzugriff pruefen
	testFile := filepath.Join(dir, ".write_test")
	if err := os.WriteFile(testFile, []byte{}, 0644); err != nil {
		return err
	}

	// Test-Datei wieder loeschen
	return os.Remove(testFile)
}

// ============================================================================
// ToLoadOptions - Konvertierung zu LoadOptions
// ============================================================================

// ToLoadOptions konvertiert VisionConfig zu LoadOptions fuer Encoder.
// Ermoeglicht nahtlose Integration mit bestehenden Encoder-Factories.
func (c *VisionConfig) ToLoadOptions() LoadOptions {
	opts := DefaultLoadOptions()

	// Threads uebernehmen
	opts.Threads = c.Threads

	// BatchSize uebernehmen
	opts.BatchSize = c.MaxBatchSize

	// Device basierend auf PreferGPU setzen
	if c.PreferGPU {
		// Plattform-spezifische GPU-Auswahl
		switch runtime.GOOS {
		case "darwin":
			opts.Device = DeviceMetal
		default:
			opts.Device = DeviceCUDA
		}
	} else {
		opts.Device = DeviceCPU
	}

	return opts
}

// ============================================================================
// String - Debug-Ausgabe
// ============================================================================

// String gibt eine lesbare Darstellung der Konfiguration zurueck.
func (c *VisionConfig) String() string {
	return "VisionConfig{Backend: " + c.Backend +
		", PreferGPU: " + strconv.FormatBool(c.PreferGPU) +
		", CacheDir: " + c.CacheDir +
		", DefaultModel: " + c.DefaultModel +
		", MaxBatchSize: " + strconv.Itoa(c.MaxBatchSize) +
		", Threads: " + strconv.Itoa(c.Threads) + "}"
}
