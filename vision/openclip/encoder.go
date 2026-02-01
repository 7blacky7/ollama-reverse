// MODUL: openclip/encoder
// ZWECK: Go-Bindings fuer LAION OpenCLIP Vision Encoder via CGO
// INPUT: Modell-Pfad (GGUF), Bild-Daten (JPEG/PNG), LoadOptions
// OUTPUT: Float32 Embeddings, ModelInfo
// NEBENEFFEKTE: Laedt OpenCLIP-Modell, alloziert C-Speicher
// ABHAENGIGKEITEN: vision (Interface), openclip.h (C-Bindings)
// HINWEISE: Unterstuetzt groessere Modelle wie ViT-bigG-14, ViT-H-14

package openclip

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lopenclip
#include "openclip.h"
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
	ErrNullContext  = errors.New("openclip: null context")
	ErrNullImage    = errors.New("openclip: null image data")
	ErrDecodeFailed = errors.New("openclip: image decode failed")
	ErrEncodeFailed = errors.New("openclip: encoding failed")
	ErrAllocFailed  = errors.New("openclip: memory allocation failed")
	ErrClosed       = errors.New("openclip: encoder already closed")
)

// ============================================================================
// OpenCLIPEncoder - Hauptstruktur
// ============================================================================

// OpenCLIPEncoder implementiert vision.VisionEncoder fuer LAION OpenCLIP-Modelle.
// Unterstuetzt groessere Modelle wie ViT-bigG-14 und ViT-H-14.
type OpenCLIPEncoder struct {
	ctx  *C.openclip_ctx
	info vision.ModelInfo
}

// ============================================================================
// Konstruktor - NewOpenCLIPEncoder
// ============================================================================

// NewOpenCLIPEncoder erstellt einen neuen OpenCLIP Encoder aus einer GGUF-Datei.
func NewOpenCLIPEncoder(modelPath string, opts vision.LoadOptions) (*OpenCLIPEncoder, error) {
	// C-Parameter vorbereiten
	params := buildCParams(opts)

	// Modell-Pfad als C-String
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	// OpenCLIP-Modell laden
	ctx := C.openclip_init(cPath, params)
	if ctx == nil {
		return nil, ErrNullContext
	}

	// Modell-Info extrahieren
	info := extractModelInfo(ctx)

	return &OpenCLIPEncoder{
		ctx:  ctx,
		info: info,
	}, nil
}

// ============================================================================
// VisionEncoder Interface - Encode
// ============================================================================

// Encode konvertiert ein einzelnes Bild zu einem Embedding-Vektor.
func (e *OpenCLIPEncoder) Encode(imageData []byte) ([]float32, error) {
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
	result := C.openclip_encode_image(
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
func (e *OpenCLIPEncoder) EncodeBatch(images [][]byte) ([][]float32, error) {
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
	result := C.openclip_encode_batch(
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

// Close gibt den OpenCLIP-Context und zugehoerigen Speicher frei.
func (e *OpenCLIPEncoder) Close() error {
	if e.ctx == nil {
		return ErrClosed
	}

	C.openclip_free(e.ctx)
	e.ctx = nil
	return nil
}

// ModelInfo gibt Metadaten ueber das geladene Modell zurueck.
func (e *OpenCLIPEncoder) ModelInfo() vision.ModelInfo {
	return e.info
}

// ============================================================================
// Hilfsfunktionen - Parameter-Konvertierung
// ============================================================================

// buildCParams konvertiert LoadOptions zu C-Parametern.
func buildCParams(opts vision.LoadOptions) C.openclip_init_params {
	params := C.openclip_default_params()
	params.n_threads = C.int32_t(opts.Threads)
	params.n_gpu_layers = C.int32_t(opts.GPULayers)
	params.main_gpu = C.int32_t(opts.MainGPU)
	params.use_mmap = boolToInt8(opts.UseMmap)
	params.use_mlock = boolToInt8(opts.UseMlock)
	return params
}

// extractModelInfo holt Modell-Metadaten aus dem C-Context.
func extractModelInfo(ctx *C.openclip_ctx) vision.ModelInfo {
	cInfo := C.openclip_get_model_info(ctx)
	return vision.ModelInfo{
		Name:         C.GoString(cInfo.name),
		Type:         "openclip",
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
