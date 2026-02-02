// MODUL: encoder_gguf
// ZWECK: Nomic Embed Vision GGUF Encoder - CGO Bindings fuer nomic.h
// INPUT: Modell-Pfad (GGUF), Bild-Pfade (JPEG, PNG)
// OUTPUT: Embedding-Vektoren ([]float32, 768-dim)
// NEBENEFFEKTE: Laedt C-Library, alloziert nativen Speicher
// ABHAENGIGKEITEN: nomic_cpp/nomic.h, vision (LoadImage)
// HINWEISE: Pfad-basiertes Interface, Close() MUSS aufgerufen werden

//go:build vision && cgo && nomic_gguf

package nomic

/*
#cgo CFLAGS: -I${SRCDIR}/../nomic_cpp
#cgo LDFLAGS: -L${SRCDIR}/../nomic_cpp -lnomic -lstdc++ -lm
#include "nomic.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"image"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/vision"
)

// Fehler-Definitionen
var (
	ErrGGUFModelLoad = errors.New("nomic_gguf: modell laden fehlgeschlagen")
	ErrGGUFNotLoaded = errors.New("nomic_gguf: modell nicht geladen")
	ErrGGUFEncode    = errors.New("nomic_gguf: encoding fehlgeschlagen")
	ErrGGUFImageLoad = errors.New("nomic_gguf: bild laden fehlgeschlagen")
)

// NomicGGUFEncoder implementiert VisionEncoder mit GGUF-Backend.
// Verwendet nomic_cpp C-Library fuer die Inferenz.
type NomicGGUFEncoder struct {
	ctx       *C.nomic_ctx
	modelPath string
	loaded    bool
	mu        sync.RWMutex
}

// NewNomicGGUFEncoder erstellt einen neuen Encoder (ohne Modell zu laden).
func NewNomicGGUFEncoder() *NomicGGUFEncoder {
	return &NomicGGUFEncoder{loaded: false}
}

// Load laedt das GGUF-Modell.
func (e *NomicGGUFEncoder) Load(modelPath string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.loaded {
		C.nomic_free(e.ctx)
		e.ctx = nil
		e.loaded = false
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	ctx := C.nomic_init(cPath, 0) // 0 = Auto-Detect Threads
	if ctx == nil {
		return fmt.Errorf("%w: %s", ErrGGUFModelLoad, getCError())
	}

	e.ctx = ctx
	e.modelPath = modelPath
	e.loaded = true
	return nil
}

// Close gibt alle Ressourcen frei.
func (e *NomicGGUFEncoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return nil
	}
	if e.ctx != nil {
		C.nomic_free(e.ctx)
		e.ctx = nil
	}
	e.loaded = false
	return nil
}

// Encode konvertiert ein Bild (Pfad) zu einem Embedding-Vektor.
func (e *NomicGGUFEncoder) Encode(imagePath string) ([]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.loaded {
		return nil, ErrGGUFNotLoaded
	}

	// Bild laden und auf 384x384 skalieren
	img, err := vision.LoadImage(imagePath)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrGGUFImageLoad, err)
	}
	resized, err := vision.ResizeImage(img, DefaultImageSize, DefaultImageSize)
	if err != nil {
		return nil, fmt.Errorf("%w: resize: %v", ErrGGUFImageLoad, err)
	}

	// RGB-Daten extrahieren und C-API aufrufen
	rgbData := extractRGB(resized.Image)
	cData := (*C.uint8_t)(unsafe.Pointer(&rgbData[0]))
	emb := C.nomic_encode_image(e.ctx, cData, C.int(DefaultImageSize), C.int(DefaultImageSize))
	if emb == nil {
		return nil, fmt.Errorf("%w: %s", ErrGGUFEncode, getCError())
	}
	defer C.nomic_embedding_free(emb)

	// Ergebnis kopieren
	dim := int(emb.dim)
	result := make([]float32, dim)
	copy(result, unsafe.Slice((*float32)(unsafe.Pointer(emb.data)), dim))
	return result, nil
}

// EncodeBatch konvertiert mehrere Bilder zu Embedding-Vektoren.
func (e *NomicGGUFEncoder) EncodeBatch(paths []string) ([][]float32, error) {
	results := make([][]float32, len(paths))
	for i, path := range paths {
		emb, err := e.Encode(path)
		if err != nil {
			return nil, fmt.Errorf("batch[%d]: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

// ModelInfo gibt Metadaten ueber das Modell zurueck.
func (e *NomicGGUFEncoder) ModelInfo() vision.ModelInfo {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.loaded || e.ctx == nil {
		return vision.ModelInfo{Name: "nomic-gguf", Type: "nomic"}
	}
	return vision.ModelInfo{
		Name:         "nomic-embed-vision-gguf",
		Type:         "nomic",
		EmbeddingDim: int(C.nomic_get_embedding_dim(e.ctx)),
		ImageSize:    int(C.nomic_get_image_size(e.ctx)),
	}
}

// extractRGB extrahiert RGB-Daten aus RGBA-Bild (HWC Format).
func extractRGB(img *image.RGBA) []byte {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgb := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcIdx := y*img.Stride + x*4
			dstIdx := (y*w + x) * 3
			rgb[dstIdx] = img.Pix[srcIdx]
			rgb[dstIdx+1] = img.Pix[srcIdx+1]
			rgb[dstIdx+2] = img.Pix[srcIdx+2]
		}
	}
	return rgb
}

// getCError holt die letzte Fehlermeldung aus der C-API.
func getCError() string {
	cErr := C.nomic_get_last_error()
	if cErr == nil {
		return "unbekannter fehler"
	}
	return C.GoString(cErr)
}
