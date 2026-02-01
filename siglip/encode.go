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
// Model Encoding - Methoden zur Bildkodierung
// ============================================================================
//
// Dieses Modul enthaelt:
// - Encode: Einzelbild-Encoding aus Bytes (JPG, PNG, etc.)
// - EncodeRaw: Encoding aus RGB-Rohdaten
// - EncodeBatch: Batch-Encoding mehrerer Bilder
// - BatchEncode: Convenience-Wrapper mit Progress-Callback

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

	// Bild aus Speicher laden (via Base64-Workaround in utils.go)
	cImg := loadImageFromMemory(image)
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

	// Erwartete Groesse berechnen (RGB = 3 Kanaele)
	const rgbChannels = 3
	expectedSize := width * height * rgbChannels
	if len(data) < expectedSize {
		return nil, fmt.Errorf("siglip: image data too small (expected %d, got %d)", expectedSize, len(data))
	}

	// C-Bild erstellen
	cImg := C.siglip_image_from_raw(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.int(width),
		C.int(height),
		C.int(rgbChannels),
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
		cImg := loadImageFromMemory(img)
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
	embeddings := extractBatchEmbeddings(cEmb, len(images))

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return embeddings, nil
}

// extractBatchEmbeddings extrahiert Embeddings aus einem C-Batch-Ergebnis
func extractBatchEmbeddings(cEmb *C.struct_siglip_embedding, numImages int) []*Embedding {
	embeddings := make([]*Embedding, numImages)
	batchSize := int(cEmb.batch_size)
	embSize := int(cEmb.size)

	// Groesse eines float32 in Bytes
	const float32Size = 4

	for i := 0; i < batchSize && i < numImages; i++ {
		emb := &Embedding{
			data:       make([]float32, embSize),
			normalized: bool(cEmb.normalized),
		}

		// Daten kopieren
		srcPtr := unsafe.Pointer(uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*embSize*float32Size))
		for j := 0; j < embSize; j++ {
			emb.data[j] = *(*float32)(unsafe.Pointer(uintptr(srcPtr) + uintptr(j*float32Size)))
		}

		embeddings[i] = emb
	}

	return embeddings
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
