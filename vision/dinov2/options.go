// MODUL: options
// ZWECK: DINOv2-spezifische Konfigurationsoptionen
// INPUT: Optionale Parameter (OutputMode, Normalisierung)
// OUTPUT: DINOv2Options Struct
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine
// HINWEISE: DINOv2 ist rein self-supervised (keine Text-Embeddings)
//           Unterstuetzt verschiedene Output-Modi (CLS, Patches, Mean)

package dinov2

// ============================================================================
// Konstanten
// ============================================================================

const (
	// EncoderName ist der Registry-Name fuer diesen Encoder
	EncoderName = "dinov2"

	// DefaultEmbeddingDim ist die Standard-Embedding-Dimension
	// DINOv2-Small: 384, DINOv2-Base: 768, DINOv2-Large: 1024, DINOv2-Giant: 1536
	DefaultEmbeddingDim = 768

	// DefaultImageSize ist die Standard-Bildgroesse (518x518 fuer DINOv2)
	DefaultImageSize = 518

	// DefaultPatchSize ist die Patch-Groesse des Vision Transformers
	DefaultPatchSize = 14

	// NumPatches berechnet sich als (ImageSize/PatchSize)^2 = 37^2 = 1369
	NumPatches = 1369
)

// ============================================================================
// OutputMode - DINOv2 Output-Varianten
// ============================================================================

// OutputMode definiert welche Features DINOv2 zurueckgibt.
type OutputMode int

const (
	// OutputCLS gibt nur das CLS Token zurueck (1 x EmbedDim)
	OutputCLS OutputMode = iota

	// OutputPatches gibt alle Patch Tokens zurueck (NumPatches x EmbedDim)
	OutputPatches

	// OutputMean gibt den Durchschnitt aller Patches zurueck (1 x EmbedDim)
	OutputMean
)

// String gibt eine lesbare Darstellung des OutputModes zurueck.
func (m OutputMode) String() string {
	switch m {
	case OutputCLS:
		return "cls"
	case OutputPatches:
		return "patches"
	case OutputMean:
		return "mean"
	default:
		return "unknown"
	}
}

// ============================================================================
// DINOv2Options - Encoder-spezifische Optionen
// ============================================================================

// DINOv2Options enthaelt DINOv2-spezifische Konfiguration.
// Erweitert die generischen vision.LoadOptions.
type DINOv2Options struct {
	// OutputMode definiert welche Features extrahiert werden
	OutputMode OutputMode

	// Normalize aktiviert L2-Normalisierung der Embeddings
	Normalize bool

	// ReturnAttention gibt auch Attention Maps zurueck (optional)
	ReturnAttention bool

	// RegisterTokens aktiviert Register-Tokens (DINOv2 mit Registern)
	RegisterTokens bool
}

// ============================================================================
// Default-Optionen
// ============================================================================

// DefaultDINOv2Options gibt die Standard-Konfiguration zurueck.
func DefaultDINOv2Options() DINOv2Options {
	return DINOv2Options{
		OutputMode:      OutputCLS, // CLS Token als Default
		Normalize:       true,      // L2-Normalisierung empfohlen
		ReturnAttention: false,     // Keine Attention Maps
		RegisterTokens:  false,     // Standard DINOv2 ohne Register
	}
}

// ============================================================================
// Option Builder
// ============================================================================

// DINOv2Option ist eine funktionale Option fuer DINOv2Options.
type DINOv2Option func(*DINOv2Options)

// WithOutputMode setzt den Output-Modus.
func WithOutputMode(mode OutputMode) DINOv2Option {
	return func(o *DINOv2Options) {
		o.OutputMode = mode
	}
}

// WithNormalize setzt die L2-Normalisierung.
func WithNormalize(enabled bool) DINOv2Option {
	return func(o *DINOv2Options) {
		o.Normalize = enabled
	}
}

// WithReturnAttention aktiviert/deaktiviert Attention Maps.
func WithReturnAttention(enabled bool) DINOv2Option {
	return func(o *DINOv2Options) {
		o.ReturnAttention = enabled
	}
}

// WithRegisterTokens aktiviert/deaktiviert Register-Tokens.
func WithRegisterTokens(enabled bool) DINOv2Option {
	return func(o *DINOv2Options) {
		o.RegisterTokens = enabled
	}
}

// Apply wendet alle Optionen an.
func (o *DINOv2Options) Apply(opts ...DINOv2Option) {
	for _, opt := range opts {
		opt(o)
	}
}
