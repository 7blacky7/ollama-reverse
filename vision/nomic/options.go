// MODUL: options
// ZWECK: Nomic-spezifische Konfigurationsoptionen
// INPUT: Optionale Parameter (Normalisierung, Pooling)
// OUTPUT: NomicOptions Struct
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: Erweiterung der generischen vision.LoadOptions
//           fuer Nomic-spezifische Features

package nomic

// ============================================================================
// Konstanten
// ============================================================================

const (
	// EncoderName ist der Registry-Name fuer diesen Encoder
	EncoderName = "nomic"

	// DefaultEmbeddingDim ist die Standard-Embedding-Dimension (768)
	DefaultEmbeddingDim = 768

	// DefaultImageSize ist die Standard-Bildgroesse (384x384)
	DefaultImageSize = 384

	// DefaultPatchSize ist die Patch-Groesse des Vision Transformers
	DefaultPatchSize = 14
)

// ============================================================================
// NomicOptions - Encoder-spezifische Optionen
// ============================================================================

// NomicOptions enthaelt Nomic-spezifische Konfiguration.
// Erweitert die generischen vision.LoadOptions.
type NomicOptions struct {
	// Normalize aktiviert L2-Normalisierung der Embeddings
	Normalize bool

	// PoolingMode definiert wie Patch-Embeddings kombiniert werden
	// "cls" = CLS Token, "mean" = Mean Pooling
	PoolingMode string

	// ProjectDim reduziert die Dimension (0 = keine Projektion)
	ProjectDim int
}

// ============================================================================
// Default-Optionen
// ============================================================================

// DefaultNomicOptions gibt die Standard-Konfiguration zurueck.
func DefaultNomicOptions() NomicOptions {
	return NomicOptions{
		Normalize:   true,  // Nomic empfiehlt normalisierte Embeddings
		PoolingMode: "cls", // CLS Token als Default
		ProjectDim:  0,     // Keine Dimensionsreduktion
	}
}

// ============================================================================
// Option Builder
// ============================================================================

// NomicOption ist eine funktionale Option fuer NomicOptions.
type NomicOption func(*NomicOptions)

// WithNormalize setzt die L2-Normalisierung.
func WithNormalize(enabled bool) NomicOption {
	return func(o *NomicOptions) {
		o.Normalize = enabled
	}
}

// WithPoolingMode setzt den Pooling-Modus.
// Gueltige Werte: "cls", "mean"
func WithPoolingMode(mode string) NomicOption {
	return func(o *NomicOptions) {
		o.PoolingMode = mode
	}
}

// WithProjectDim setzt die Projektiondimension.
// 0 deaktiviert die Projektion.
func WithProjectDim(dim int) NomicOption {
	return func(o *NomicOptions) {
		if dim >= 0 {
			o.ProjectDim = dim
		}
	}
}

// Apply wendet alle Optionen an.
func (o *NomicOptions) Apply(opts ...NomicOption) {
	for _, opt := range opts {
		opt(o)
	}
}
