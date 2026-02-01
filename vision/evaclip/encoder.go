// MODUL: evaclip/encoder
// ZWECK: Go-Bindings fuer EVA-CLIP Vision Encoder via CGO
// INPUT: Modell-Pfad (GGUF), Bild-Daten (JPEG/PNG), LoadOptions
// OUTPUT: Float32 Embeddings, ModelInfo
// NEBENEFFEKTE: Laedt EVA-CLIP-Modell, alloziert C-Speicher
// ABHAENGIGKEITEN: vision (Interface), evaclip.h (C-Bindings)
// HINWEISE: EVA02-CLIP nutzt EVA-Initialisierung fuer bessere Repr. Close() muss aufgerufen werden.

package evaclip

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -levaclip
#include "evaclip.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"unsafe"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Fehler-Definitionen
// ============================================================================

var (
	ErrNullContext  = errors.New("evaclip: null context")
	ErrNullImage    = errors.New("evaclip: null image data")
	ErrDecodeFailed = errors.New("evaclip: image decode failed")
	ErrEncodeFailed = errors.New("evaclip: encoding failed")
	ErrAllocFailed  = errors.New("evaclip: memory allocation failed")
	ErrClosed       = errors.New("evaclip: encoder already closed")
)

// ============================================================================
// EVACLIPEncoder - Hauptstruktur
// ============================================================================

// EVACLIPEncoder implementiert vision.VisionEncoder fuer EVA02-CLIP-Modelle.
// EVA-CLIP nutzt die EVA-Initialisierung (aus MAE) fuer verbesserte Features.
type EVACLIPEncoder struct {
	ctx  *C.evaclip_ctx
	info vision.ModelInfo
}

// ============================================================================
// Konstruktor - NewEVACLIPEncoder
// ============================================================================

// NewEVACLIPEncoder erstellt einen neuen EVA-CLIP Encoder aus einer GGUF-Datei.
func NewEVACLIPEncoder(modelPath string, opts vision.LoadOptions) (*EVACLIPEncoder, error) {
	// C-Parameter vorbereiten
	params := buildCParams(opts)

	// Modell-Pfad als C-String
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	// EVA-CLIP-Modell laden
	ctx := C.evaclip_init(cPath, params)
	if ctx == nil {
		return nil, ErrNullContext
	}

	// Modell-Info extrahieren
	info := extractModelInfo(ctx)

	return &EVACLIPEncoder{
		ctx:  ctx,
		info: info,
	}, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode konvertiert ein einzelnes Bild zu einem Embedding-Vektor.
func (e *EVACLIPEncoder) Encode(imageData []byte) ([]float32, error) {
	if e.ctx == nil {
		return nil, ErrClosed
	}
	if len(imageData) == 0 {
		return nil, ErrNullImage
	}

	// Embedding-Buffer allozieren
	embeddingDim := e.info.EmbeddingDim
	embedding := make([]float32, embeddingDim)

	// C-Funktion aufrufen
	result := C.evaclip_encode_image(
		e.ctx,
		(*C.uint8_t)(unsafe.Pointer(&imageData[0])),
		C.size_t(len(imageData)),
		(*C.float)(unsafe.Pointer(&embedding[0])),
		C.int32_t(embeddingDim),
	)

	// Fehlercode auswerten
	if err := mapCError(int(result)); err != nil {
		return nil, err
	}

	return embedding, nil
}

// ============================================================================
// VisionEncoder Interface - EncodeBatch
// ============================================================================

// EncodeBatch konvertiert mehrere Bilder zu Embedding-Vektoren.
func (e *EVACLIPEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
	if e.ctx == nil {
		return nil, ErrClosed
	}
	if len(images) == 0 {
		return [][]float32{}, nil
	}

	batchSize := len(images)
	embeddingDim := e.info.EmbeddingDim

	// C-Arrays vorbereiten
	cImages, cSizes, err := prepareBatchArrays(images)
	if err != nil {
		return nil, err
	}
	defer freeBatchArrays(cImages, batchSize)

	// Flaches Embedding-Array fuer C
	flatEmbeddings := make([]float32, batchSize*embeddingDim)

	// C-Funktion aufrufen
	result := C.evaclip_encode_batch(
		e.ctx,
		(**C.uint8_t)(unsafe.Pointer(&cImages[0])),
		(*C.size_t)(unsafe.Pointer(&cSizes[0])),
		C.int32_t(batchSize),
		(*C.float)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int32_t(embeddingDim),
	)

	if err := mapCError(int(result)); err != nil {
		return nil, err
	}

	// Flaches Array zu 2D konvertieren
	return reshapeEmbeddings(flatEmbeddings, batchSize, embeddingDim), nil
}

// ============================================================================
// VisionEncoder Interface - Close & ModelInfo
// ============================================================================

// Close gibt den EVA-CLIP-Context und zugehoerigen Speicher frei.
func (e *EVACLIPEncoder) Close() error {
	if e.ctx == nil {
		return ErrClosed
	}

	C.evaclip_free(e.ctx)
	e.ctx = nil
	return nil
}

// ModelInfo gibt Metadaten ueber das geladene Modell zurueck.
func (e *EVACLIPEncoder) ModelInfo() vision.ModelInfo {
	return e.info
}

// ============================================================================
// Hilfsfunktionen - Parameter-Konvertierung
// ============================================================================

// buildCParams konvertiert LoadOptions zu C-Parametern.
func buildCParams(opts vision.LoadOptions) C.evaclip_init_params {
	params := C.evaclip_default_params()
	params.n_threads = C.int32_t(opts.Threads)
	params.n_gpu_layers = C.int32_t(opts.GPULayers)
	params.main_gpu = C.int32_t(opts.MainGPU)
	params.use_mmap = boolToInt8(opts.UseMmap)
	params.use_mlock = boolToInt8(opts.UseMlock)
	return params
}

// extractModelInfo holt Modell-Metadaten aus dem C-Context.
func extractModelInfo(ctx *C.evaclip_ctx) vision.ModelInfo {
	cInfo := C.evaclip_get_model_info(ctx)
	return vision.ModelInfo{
		Name:         C.GoString(cInfo.name),
		Type:         "evaclip",
		EmbeddingDim: int(cInfo.embedding_dim),
		ImageSize:    int(cInfo.image_size),
	}
}

// ============================================================================
// Hilfsfunktionen - Batch-Verarbeitung
// ============================================================================

// prepareBatchArrays erstellt C-Arrays fuer Batch-Encoding.
func prepareBatchArrays(images [][]byte) ([]*C.uint8_t, []C.size_t, error) {
	batchSize := len(images)
	cImages := make([]*C.uint8_t, batchSize)
	cSizes := make([]C.size_t, batchSize)

	for i, img := range images {
		if len(img) == 0 {
			freeBatchArrays(cImages, i)
			return nil, nil, ErrNullImage
		}
		// Kopie fuer C-Seite erstellen
		cImages[i] = (*C.uint8_t)(C.CBytes(img))
		cSizes[i] = C.size_t(len(img))
	}

	return cImages, cSizes, nil
}

// freeBatchArrays gibt allozierte C-Speicher frei.
func freeBatchArrays(cImages []*C.uint8_t, count int) {
	for i := 0; i < count; i++ {
		if cImages[i] != nil {
			C.free(unsafe.Pointer(cImages[i]))
		}
	}
}

// reshapeEmbeddings konvertiert flaches Array zu 2D-Slice.
func reshapeEmbeddings(flat []float32, batchSize, dim int) [][]float32 {
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * dim
		result[i] = flat[start : start+dim]
	}
	return result
}

// ============================================================================
// Hilfsfunktionen - Fehler-Mapping
// ============================================================================

// mapCError konvertiert C-Fehlercodes zu Go-Errors.
func mapCError(code int) error {
	switch code {
	case 0:
		return nil
	case -1:
		return ErrNullContext
	case -2:
		return ErrNullImage
	case -3:
		return ErrDecodeFailed
	case -4:
		return ErrEncodeFailed
	case -5:
		return ErrAllocFailed
	default:
		return ErrEncodeFailed
	}
}

// boolToInt8 konvertiert bool zu C.int8_t.
func boolToInt8(b bool) C.int8_t {
	if b {
		return 1
	}
	return 0
}
