package siglip

/*
#include <stdlib.h>
#include "siglip.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ============================================================================
// Embedding
// ============================================================================

// Embedding repraesentiert ein SigLIP Image-Embedding
type Embedding struct {
	data       []float32
	normalized bool
}

// newEmbeddingFromC erstellt ein Go-Embedding aus einem C-Embedding
func newEmbeddingFromC(cEmb *C.struct_siglip_embedding) *Embedding {
	if cEmb == nil {
		return nil
	}

	size := int(cEmb.size)
	emb := &Embedding{
		data:       make([]float32, size),
		normalized: bool(cEmb.normalized),
	}

	// Daten kopieren
	for i := 0; i < size; i++ {
		emb.data[i] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*4))))
	}

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return emb
}

// ToFloat32 gibt das Embedding als float32-Slice zurueck
func (e *Embedding) ToFloat32() []float32 {
	if e == nil {
		return nil
	}
	// Kopie zurueckgeben
	result := make([]float32, len(e.data))
	copy(result, e.data)
	return result
}

// Size gibt die Dimension des Embeddings zurueck
func (e *Embedding) Size() int {
	if e == nil {
		return 0
	}
	return len(e.data)
}

// IsNormalized gibt zurueck ob das Embedding L2-normalisiert ist
func (e *Embedding) IsNormalized() bool {
	if e == nil {
		return false
	}
	return e.normalized
}

// Normalize normalisiert das Embedding in-place (L2-Norm)
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

// CosineSimilarity berechnet die Cosine Similarity zwischen zwei Embeddings
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

// DotProduct berechnet das Skalarprodukt zwischen zwei Embeddings
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

// Clone erstellt eine Kopie des Embeddings
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
// Model Encoding Methods
// ============================================================================

// Encode generiert ein Embedding fuer ein Bild (Rohdaten)
//
// Das Bild wird als Byte-Array uebergeben (JPG, PNG, etc.)
func (m *Model) Encode(image []byte) (*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(image) == 0 {
		return nil, ErrInvalidImage
	}

	// Bild aus Speicher laden (siglip_image_from_raw erwartet RGB-Daten,
	// wir muessen stb_image nutzen - das passiert in C)
	// Da siglip.h keine direkte Funktion fuer Speicher-Bilder hat,
	// nutzen wir einen Workaround ueber Base64 oder temporaere Datei.
	// Hier implementieren wir es ueber direkte Raw-Daten nach Dekodierung.

	// Fuer jetzt: Wir erwarten RGB-Daten direkt oder implementieren
	// eine Hilfsfunktion. Da siglip_image_from_base64 existiert, nutzen wir das.
	// Besser: Wir fuegen eine neue C-Funktion hinzu oder nutzen stb_image direkt.

	// Temporaerer Workaround: Wir nehmen an, dass das Bild bereits dekodiert ist
	// als RGB uint8 Array mit bekannten Dimensionen.
	// In der Praxis wuerde man hier stb_image_load_from_memory nutzen.

	// Fuer eine vollstaendige Implementation fuegen wir eine Hilfsfunktion hinzu:
	cImg := m.loadImageFromMemory(image)
	if cImg == nil {
		return nil, ErrInvalidImage
	}
	defer C.siglip_image_free(cImg)

	// Encoding
	cEmb := C.siglip_encode(m.ctx, cImg)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	return newEmbeddingFromC(cEmb), nil
}

// EncodeRaw generiert ein Embedding fuer ein Bild aus RGB-Rohdaten
//
// data: RGB uint8 Array (HWC Format)
// width, height: Bildgroesse
func (m *Model) EncodeRaw(data []byte, width, height int) (*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(data) == 0 || width <= 0 || height <= 0 {
		return nil, ErrInvalidImage
	}

	expectedSize := width * height * 3
	if len(data) < expectedSize {
		return nil, fmt.Errorf("siglip: image data too small (expected %d, got %d)", expectedSize, len(data))
	}

	// C-Bild erstellen
	cImg := C.siglip_image_from_raw(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.int(width),
		C.int(height),
		C.int(3), // channels
	)
	if cImg == nil {
		return nil, ErrInvalidImage
	}
	defer C.siglip_image_free(cImg)

	// Encoding
	cEmb := C.siglip_encode(m.ctx, cImg)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	return newEmbeddingFromC(cEmb), nil
}

// EncodeBatch generiert Embeddings fuer mehrere Bilder
func (m *Model) EncodeBatch(images [][]byte) ([]*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(images) == 0 {
		return nil, ErrInvalidParameters
	}

	// C-Bilder erstellen
	cImages := make([]*C.struct_siglip_image, len(images))
	for i, img := range images {
		cImg := m.loadImageFromMemory(img)
		if cImg == nil {
			// Cleanup bereits erstellte Bilder
			for j := 0; j < i; j++ {
				C.siglip_image_free(cImages[j])
			}
			return nil, fmt.Errorf("siglip: failed to load image %d", i)
		}
		cImages[i] = cImg
	}

	// Cleanup am Ende
	defer func() {
		for _, cImg := range cImages {
			if cImg != nil {
				C.siglip_image_free(cImg)
			}
		}
	}()

	// Batch erstellen
	var batch C.struct_siglip_batch
	batch.images = (**C.struct_siglip_image)(unsafe.Pointer(&cImages[0]))
	batch.n_images = C.int(len(images))

	// Batch-Encoding
	cEmb := C.siglip_encode_batch(m.ctx, &batch)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	// Embeddings extrahieren
	embeddings := make([]*Embedding, len(images))
	batchSize := int(cEmb.batch_size)
	embSize := int(cEmb.size)

	for i := 0; i < batchSize && i < len(images); i++ {
		emb := &Embedding{
			data:       make([]float32, embSize),
			normalized: bool(cEmb.normalized),
		}

		// Daten kopieren
		srcPtr := unsafe.Pointer(uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*embSize*4))
		for j := 0; j < embSize; j++ {
			emb.data[j] = *(*float32)(unsafe.Pointer(uintptr(srcPtr) + uintptr(j*4)))
		}

		embeddings[i] = emb
	}

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return embeddings, nil
}

// loadImageFromMemory laedt ein Bild aus Speicher
// Dies ist eine Hilfsfunktion, die stb_image nutzt
func (m *Model) loadImageFromMemory(data []byte) *C.struct_siglip_image {
	// Da siglip.h keine direkte Funktion fuer Speicher-Bilder hat,
	// implementieren wir einen Workaround.
	// Option 1: Base64-Encoding (ineffizient)
	// Option 2: Temporaere Datei (langsam)
	// Option 3: Direkte stb_image Nutzung in CGO

	// Fuer die vollstaendige Implementation wuerde man eine neue C-Funktion
	// siglip_image_from_memory hinzufuegen. Hier nutzen wir einen Workaround
	// mit Base64.

	// Base64 encoding
	base64Data := base64Encode(data)
	cBase64 := C.CString(base64Data)
	defer C.free(unsafe.Pointer(cBase64))

	return C.siglip_image_from_base64(cBase64)
}

// BatchEncode ist ein Convenience-Wrapper fuer Batch-Encoding mit Fehlerbehandlung
func (m *Model) BatchEncode(images [][]byte, onProgress func(current, total int)) ([]*Embedding, []error) {
	embeddings := make([]*Embedding, len(images))
	errors := make([]error, len(images))

	for i, img := range images {
		emb, err := m.Encode(img)
		embeddings[i] = emb
		errors[i] = err

		if onProgress != nil {
			onProgress(i+1, len(images))
		}
	}

	return embeddings, errors
}
