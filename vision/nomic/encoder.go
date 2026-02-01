// MODUL: encoder
// ZWECK: Nomic Embed Vision Encoder - Go Bindings mit CGO
// INPUT: Modell-Pfad, Bild-Daten ([]byte), LoadOptions
// OUTPUT: Embedding-Vektoren ([]float32), ModelInfo
// NEBENEFFEKTE: Laedt C-Library, alloziert nativen Speicher
// ABHAENGIGKEITEN: vision (VisionEncoder Interface), nomic.h (C-API)
// HINWEISE: Unified Text+Image Embedding Space, 768-dim Output
//           Thread-sicher durch CGO, Close() MUSS aufgerufen werden

package nomic

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lnomic
#include "nomic.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Fehler-Definitionen
// ============================================================================

var (
	ErrModelLoad     = errors.New("nomic: failed to load model")
	ErrNullContext   = errors.New("nomic: null context")
	ErrEncodeFailed  = errors.New("nomic: encoding failed")
	ErrAlreadyClosed = errors.New("nomic: encoder already closed")
)

// ============================================================================
// NomicEncoder - Hauptstruktur
// ============================================================================

// NomicEncoder implementiert vision.VisionEncoder fuer Nomic Embed Vision.
// Besonderheit: Unified Text+Image Embedding Space (768-dim).
type NomicEncoder struct {
	ctx    *C.nomic_ctx
	info   vision.ModelInfo
	opts   NomicOptions
	closed bool
	mu     sync.RWMutex
}

// ============================================================================
// Konstruktor
// ============================================================================

// NewNomicEncoder erstellt einen neuen Nomic Embed Vision Encoder.
// modelPath: Pfad zur GGUF-Modelldatei
// loadOpts: Konfiguration (Threads, Device, etc.)
func NewNomicEncoder(modelPath string, loadOpts vision.LoadOptions) (*NomicEncoder, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	// Threads aus LoadOptions, 0 = auto
	nThreads := C.int(loadOpts.Threads)
	if loadOpts.Threads <= 0 {
		nThreads = 0
	}

	ctx := C.nomic_load_model(cPath, nThreads)
	if ctx == nil {
		return nil, ErrModelLoad
	}

	// Modell-Info aus C-API holen
	embDim := int(C.nomic_get_embedding_dim(ctx))
	imgSize := int(C.nomic_get_image_size(ctx))

	enc := &NomicEncoder{
		ctx: ctx,
		info: vision.ModelInfo{
			Name:         "nomic-embed-vision",
			Type:         "nomic",
			EmbeddingDim: embDim,
			ImageSize:    imgSize,
		},
		opts:   DefaultNomicOptions(),
		closed: false,
	}

	return enc, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode konvertiert ein Bild zu einem Embedding-Vektor.
// imageData: Rohbilddaten (JPEG, PNG, etc.)
// Rueckgabe: 768-dim float32 Vektor oder Fehler
func (e *NomicEncoder) Encode(imageData []byte) ([]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed {
		return nil, ErrAlreadyClosed
	}
	if e.ctx == nil {
		return nil, ErrNullContext
	}

	return e.encodeInternal(imageData)
}

// encodeInternal fuehrt das eigentliche Encoding durch (ohne Lock).
func (e *NomicEncoder) encodeInternal(imageData []byte) ([]float32, error) {
	embDim := e.info.EmbeddingDim
	embedding := make([]float32, embDim)

	// C-API aufrufen
	cData := (*C.uint8_t)(unsafe.Pointer(&imageData[0]))
	cLen := C.size_t(len(imageData))
	cEmb := (*C.float)(unsafe.Pointer(&embedding[0]))
	cMaxDim := C.int(embDim)

	result := C.nomic_encode(e.ctx, cData, cLen, cEmb, cMaxDim)
	if result < 0 {
		return nil, ErrEncodeFailed
	}

	return embedding, nil
}

// ============================================================================
// VisionEncoder Interface - EncodeBatch
// ============================================================================

// EncodeBatch konvertiert mehrere Bilder zu Embedding-Vektoren.
// images: Array von Rohbilddaten
// Rueckgabe: Array von 768-dim float32 Vektoren
func (e *NomicEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed {
		return nil, ErrAlreadyClosed
	}
	if e.ctx == nil {
		return nil, ErrNullContext
	}

	results := make([][]float32, len(images))
	for i, img := range images {
		emb, err := e.encodeInternal(img)
		if err != nil {
			return nil, err
		}
		results[i] = emb
	}

	return results, nil
}

// ============================================================================
// VisionEncoder Interface - Close & ModelInfo
// ============================================================================

// Close gibt alle Ressourcen frei.
// MUSS aufgerufen werden um Speicherlecks zu vermeiden.
func (e *NomicEncoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}

	if e.ctx != nil {
		C.nomic_free(e.ctx)
		e.ctx = nil
	}
	e.closed = true
	return nil
}

// ModelInfo gibt Metadaten ueber das Modell zurueck.
func (e *NomicEncoder) ModelInfo() vision.ModelInfo {
	return e.info
}
