// MODUL: encoder
// ZWECK: DINOv2 Self-Supervised Vision Encoder - Go Bindings mit CGO
// INPUT: Modell-Pfad, Bild-Daten ([]byte), LoadOptions
// OUTPUT: Embedding-Vektoren ([]float32), Patch-Embeddings ([][]float32)
// NEBENEFFEKTE: Laedt C-Library, alloziert nativen Speicher
// ABHAENGIGKEITEN: vision (VisionEncoder Interface), dinov2.h (C-API)
// HINWEISE: Rein self-supervised, KEINE Text-Embeddings
//           Unterstuetzt CLS, Patch-Tokens und Mean-Pooling
//           Thread-sicher durch CGO, Close() MUSS aufgerufen werden

package dinov2

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -ldinov2
#include "dinov2.h"
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
	ErrModelLoad     = errors.New("dinov2: failed to load model")
	ErrNullContext   = errors.New("dinov2: null context")
	ErrEncodeFailed  = errors.New("dinov2: encoding failed")
	ErrAlreadyClosed = errors.New("dinov2: encoder already closed")
	ErrInvalidMode   = errors.New("dinov2: invalid output mode")
)

// ============================================================================
// DINOv2Encoder - Hauptstruktur
// ============================================================================

// DINOv2Encoder implementiert vision.VisionEncoder fuer DINOv2.
// Besonderheit: Self-Supervised Learning, keine Text-Embeddings.
type DINOv2Encoder struct {
	ctx        *C.dinov2_ctx
	info       vision.ModelInfo
	opts       DINOv2Options
	outputMode OutputMode
	closed     bool
	mu         sync.RWMutex
}

// ============================================================================
// Konstruktor
// ============================================================================

// NewDINOv2Encoder erstellt einen neuen DINOv2 Vision Encoder.
// modelPath: Pfad zur GGUF-Modelldatei
// loadOpts: Konfiguration (Threads, Device, etc.)
func NewDINOv2Encoder(modelPath string, loadOpts vision.LoadOptions) (*DINOv2Encoder, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	// Threads aus LoadOptions, 0 = auto
	nThreads := C.int(loadOpts.Threads)
	if loadOpts.Threads <= 0 {
		nThreads = 0
	}

	// C-API: dinov2_load (nicht dinov2_load_model)
	ctx := C.dinov2_load(cPath, nThreads)
	if ctx == nil {
		return nil, ErrModelLoad
	}

	// Modell-Info aus C-API holen
	// C-API: dinov2_get_dim (nicht dinov2_get_embedding_dim)
	embDim := int(C.dinov2_get_dim(ctx))
	imgSize := int(C.dinov2_get_image_size(ctx))

	enc := &DINOv2Encoder{
		ctx: ctx,
		info: vision.ModelInfo{
			Name:         "dinov2",
			Type:         EncoderName,
			EmbeddingDim: embDim,
			ImageSize:    imgSize,
		},
		opts:       DefaultDINOv2Options(),
		outputMode: OutputCLS,
		closed:     false,
	}

	return enc, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode konvertiert ein Bild zu einem Embedding-Vektor.
// imageData: Rohbilddaten (JPEG, PNG, etc.)
// Rueckgabe: float32 Vektor (Dimension abhaengig vom OutputMode)
func (e *DINOv2Encoder) Encode(imageData []byte) ([]float32, error) {
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
func (e *DINOv2Encoder) encodeInternal(imageData []byte) ([]float32, error) {
	embDim := e.info.EmbeddingDim
	embedding := make([]float32, embDim)

	// C-API aufrufen
	cData := (*C.uint8_t)(unsafe.Pointer(&imageData[0]))
	cLen := C.size_t(len(imageData))
	cEmb := (*C.float)(unsafe.Pointer(&embedding[0]))
	// C-API: Nutzt dinov2_output_mode Enum
	cMode := C.dinov2_output_mode(e.outputMode)

	result := C.dinov2_encode(e.ctx, cData, cLen, cEmb, C.int(embDim), cMode)
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
// Rueckgabe: Array von float32 Vektoren
func (e *DINOv2Encoder) EncodeBatch(images [][]byte) ([][]float32, error) {
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
func (e *DINOv2Encoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}

	if e.ctx != nil {
		C.dinov2_free(e.ctx)
		e.ctx = nil
	}
	e.closed = true
	return nil
}

// ModelInfo gibt Metadaten ueber das Modell zurueck.
func (e *DINOv2Encoder) ModelInfo() vision.ModelInfo {
	return e.info
}

// ============================================================================
// DINOv2-spezifische Methoden
// ============================================================================

// SetOutputMode setzt den Output-Modus (CLS, Patches, Mean).
func (e *DINOv2Encoder) SetOutputMode(mode OutputMode) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.outputMode = mode
}

// GetOutputMode gibt den aktuellen Output-Modus zurueck.
func (e *DINOv2Encoder) GetOutputMode() OutputMode {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.outputMode
}

// EncodePatches extrahiert alle Patch-Embeddings aus einem Bild.
// imageData: Rohbilddaten (JPEG, PNG, etc.)
// Rueckgabe: 2D Array [NumPatches][EmbedDim] float32
func (e *DINOv2Encoder) EncodePatches(imageData []byte) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed {
		return nil, ErrAlreadyClosed
	}
	if e.ctx == nil {
		return nil, ErrNullContext
	}

	return e.encodePatchesInternal(imageData)
}

// encodePatchesInternal extrahiert Patch-Embeddings (ohne Lock).
// Nutzt dinov2_encode mit DINOV2_OUTPUT_PATCHES Modus.
func (e *DINOv2Encoder) encodePatchesInternal(imageData []byte) ([][]float32, error) {
	embDim := e.info.EmbeddingDim
	numPatches := int(C.dinov2_get_num_patches(e.ctx))

	// Flaches Array fuer alle Patches (numPatches * embDim)
	totalSize := numPatches * embDim
	flatEmbeddings := make([]float32, totalSize)

	// C-API aufrufen mit PATCHES Modus
	cData := (*C.uint8_t)(unsafe.Pointer(&imageData[0]))
	cLen := C.size_t(len(imageData))
	cEmb := (*C.float)(unsafe.Pointer(&flatEmbeddings[0]))
	cMode := C.dinov2_output_mode(C.DINOV2_OUTPUT_PATCHES)

	result := C.dinov2_encode(e.ctx, cData, cLen, cEmb, C.int(totalSize), cMode)
	if result < 0 {
		return nil, ErrEncodeFailed
	}

	// In 2D Array umwandeln [numPatches][embDim]
	patches := make([][]float32, numPatches)
	for i := 0; i < numPatches; i++ {
		patches[i] = flatEmbeddings[i*embDim : (i+1)*embDim]
	}

	return patches, nil
}
