//go:build siglip
// ============================================================================
// MODUL: embedding
// ZWECK: Embedding-Datenstruktur und mathematische Operationen
// INPUT: C-Embedding-Pointer, andere Embeddings
// OUTPUT: Embedding-Vektoren ([]float32), Similarity-Scores
// NEBENEFFEKTE: Alloziert Speicher
// ABHAENGIGKEITEN: utils.go (sqrt64)
// HINWEISE: Encoding-Funktionen sind in encode.go
// ============================================================================

package siglip

/*
#include <stdlib.h>
#include "siglip.h"
*/
import "C"

import (
	"unsafe"
)

// ============================================================================
// Embedding Struct
// ============================================================================

// Embedding repraesentiert ein SigLIP Image-Embedding.
type Embedding struct {
	data       []float32 // Embedding-Vektor
	normalized bool      // L2-normalisiert?
}

// ============================================================================
// Embedding - Konstruktor aus C
// ============================================================================

// newEmbeddingFromC erstellt ein Go-Embedding aus einem C-Embedding.
// Gibt den C-Speicher automatisch frei.
func newEmbeddingFromC(cEmb *C.struct_siglip_embedding) *Embedding {
	if cEmb == nil {
		return nil
	}

	size := int(cEmb.size)
	emb := &Embedding{
		data:       make([]float32, size),
		normalized: bool(cEmb.normalized),
	}

	// Daten kopieren (float32 = 4 Bytes)
	const float32Size = 4
	for i := 0; i < size; i++ {
		emb.data[i] = float32(*(*C.float)(unsafe.Pointer(
			uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*float32Size))))
	}

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return emb
}

// ============================================================================
// Embedding - Getter-Methoden
// ============================================================================

// ToFloat32 gibt das Embedding als float32-Slice zurueck (Kopie).
func (e *Embedding) ToFloat32() []float32 {
	if e == nil {
		return nil
	}
	result := make([]float32, len(e.data))
	copy(result, e.data)
	return result
}

// Size gibt die Dimension des Embeddings zurueck.
func (e *Embedding) Size() int {
	if e == nil {
		return 0
	}
	return len(e.data)
}

// IsNormalized gibt zurueck ob das Embedding L2-normalisiert ist.
func (e *Embedding) IsNormalized() bool {
	if e == nil {
		return false
	}
	return e.normalized
}

// Clone erstellt eine Kopie des Embeddings.
func (e *Embedding) Clone() *Embedding {
	if e == nil {
		return nil
	}
	return &Embedding{
		data:       e.ToFloat32(),
		normalized: e.normalized,
	}
}

// ============================================================================
// Embedding - Mathematische Operationen
// ============================================================================

// Normalize normalisiert das Embedding in-place (L2-Norm).
func (e *Embedding) Normalize() {
	if e == nil || len(e.data) == 0 {
		return
	}

	var norm float32
	for _, v := range e.data {
		norm += v * v
	}
	norm = float32(sqrt64(float64(norm)))

	if norm > 0 {
		for i := range e.data {
			e.data[i] /= norm
		}
	}
	e.normalized = true
}

// CosineSimilarity berechnet die Cosine Similarity zwischen zwei Embeddings.
// Gibt 0 zurueck bei nil oder unterschiedlichen Dimensionen.
func (e *Embedding) CosineSimilarity(other *Embedding) float32 {
	if e == nil || other == nil {
		return 0
	}

	if len(e.data) != len(other.data) {
		return 0
	}

	var dot, normA, normB float32
	for i := range e.data {
		dot += e.data[i] * other.data[i]
		normA += e.data[i] * e.data[i]
		normB += other.data[i] * other.data[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (float32(sqrt64(float64(normA))) * float32(sqrt64(float64(normB))))
}

// DotProduct berechnet das Skalarprodukt zwischen zwei Embeddings.
func (e *Embedding) DotProduct(other *Embedding) float32 {
	if e == nil || other == nil {
		return 0
	}

	if len(e.data) != len(other.data) {
		return 0
	}

	var dot float32
	for i := range e.data {
		dot += e.data[i] * other.data[i]
	}

	return dot
}
