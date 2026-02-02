//go:build siglip
// ============================================================================
// MODUL: utils
// ZWECK: Hilfsfunktionen, globale Funktionen und Similarity-Operationen
// INPUT: Embeddings, Bilder, Konfiguration
// OUTPUT: Similarity-Scores, System-Informationen
// NEBENEFFEKTE: Ruft C-Funktionen auf
// ABHAENGIGKEITEN: model.go (C-Bindings)
// HINWEISE: Keine Magic Numbers - alle Konstanten benannt
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
// Global Functions - Version und Build Info
// ============================================================================

// Version gibt die SigLIP-Version zurueck.
func Version() string {
	return C.GoString(C.siglip_version())
}

// BuildInfo gibt Build-Informationen zurueck.
func BuildInfo() string {
	return C.GoString(C.siglip_build_info())
}

// SystemInfo gibt System-Informationen zurueck.
func SystemInfo() string {
	return C.GoString(C.siglip_system_info())
}

// ============================================================================
// Global Functions - Backend Management
// ============================================================================

// BackendAvailable prueft ob ein Backend verfuegbar ist.
func BackendAvailable(backend Backend) bool {
	return bool(C.siglip_backend_available(C.enum_siglip_backend(backend)))
}

// AvailableBackends gibt eine Liste verfuegbarer Backends zurueck.
func AvailableBackends() []Backend {
	const maxBackends = 4
	var cBackends [maxBackends]C.enum_siglip_backend
	n := C.siglip_get_available_backends(&cBackends[0], maxBackends)

	backends := make([]Backend, int(n))
	for i := 0; i < int(n); i++ {
		backends[i] = Backend(cBackends[i])
	}

	return backends
}

// ============================================================================
// Global Functions - Logging und Error Handling
// ============================================================================

// SetLogLevel setzt das globale Log-Level.
func SetLogLevel(level LogLevel) {
	C.siglip_set_log_level(C.enum_siglip_log_level(level))
}

// GetLastError gibt den letzten Fehler zurueck.
func GetLastError() string {
	errStr := C.siglip_get_last_error()
	if errStr == nil {
		return ""
	}
	return C.GoString(errStr)
}

// ClearError loescht den letzten Fehler.
func ClearError() {
	C.siglip_clear_error()
}

// ============================================================================
// Image Loading Helper
// ============================================================================

// loadImageFromMemory laedt ein Bild aus Speicher.
// Verwendet Base64-Encoding als Workaround.
func loadImageFromMemory(data []byte) *C.struct_siglip_image {
	base64Data := base64Encode(data)
	cBase64 := C.CString(base64Data)
	defer C.free(unsafe.Pointer(cBase64))

	return C.siglip_image_from_base64(cBase64)
}

// ============================================================================
// Image Struct
// ============================================================================

// Image repraesentiert ein Bild fuer SigLIP.
type Image struct {
	Data     []byte // Bilddaten
	Width    int    // Breite in Pixel
	Height   int    // Hoehe in Pixel
	Channels int    // Anzahl Kanaele (3 = RGB)
}

// NewImageFromRGB erstellt ein neues Image aus RGB-Daten.
func NewImageFromRGB(data []byte, width, height int) *Image {
	const rgbChannels = 3
	return &Image{
		Data:     data,
		Width:    width,
		Height:   height,
		Channels: rgbChannels,
	}
}

// ============================================================================
// Similarity Functions - Matrix und Suche
// ============================================================================

// CosineSimilarityMatrix berechnet die Cosine Similarity Matrix fuer mehrere Embeddings.
// Nutzt Symmetrie zur Optimierung.
func CosineSimilarityMatrix(embeddings []*Embedding) [][]float32 {
	n := len(embeddings)
	matrix := make([][]float32, n)

	for i := range matrix {
		matrix[i] = make([]float32, n)
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0 // Selbst-Similarity ist immer 1
			} else if i < j {
				matrix[i][j] = embeddings[i].CosineSimilarity(embeddings[j])
			} else {
				matrix[i][j] = matrix[j][i] // Symmetrie nutzen
			}
		}
	}

	return matrix
}

// FindMostSimilar findet die aehnlichsten Embeddings zu einem Query-Embedding.
// Gibt die Indizes der Top-K aehnlichsten Kandidaten zurueck.
func FindMostSimilar(query *Embedding, candidates []*Embedding, topK int) []int {
	if query == nil || len(candidates) == 0 {
		return nil
	}

	// Similarities berechnen
	type scored struct {
		index int
		score float32
	}

	scores := make([]scored, len(candidates))
	for i, cand := range candidates {
		scores[i] = scored{
			index: i,
			score: query.CosineSimilarity(cand),
		}
	}

	// Sortieren (einfaches Selection Sort fuer kleine Listen)
	for i := 0; i < len(scores)-1; i++ {
		maxIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[maxIdx].score {
				maxIdx = j
			}
		}
		if maxIdx != i {
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}
	}

	// Top-K extrahieren
	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = scores[i].index
	}

	return result
}

// ============================================================================
// Helper Functions - Mathematik
// ============================================================================

// sqrt64 berechnet die Quadratwurzel mit Newton-Raphson Verfahren.
func sqrt64(x float64) float64 {
	if x <= 0 {
		return 0
	}

	// Newton-Raphson Iterationen
	const iterations = 10
	z := x
	for i := 0; i < iterations; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// ============================================================================
// Helper Functions - Base64 Encoding
// ============================================================================

// base64Encode kodiert Bytes zu Base64 (RFC 4648).
func base64Encode(data []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

	// Ergebnis-Buffer allozieren (ceil(len/3) * 4)
	resultLen := (len(data) + 2) / 3 * 4
	result := make([]byte, 0, resultLen)

	// Je 3 Bytes zu 4 Base64-Zeichen konvertieren
	const bytesPerGroup = 3
	for i := 0; i < len(data); i += bytesPerGroup {
		var n uint32

		remaining := len(data) - i
		switch remaining {
		case 1:
			// 1 Byte: xx==
			n = uint32(data[i]) << 16
			result = append(result,
				alphabet[n>>18],
				alphabet[(n>>12)&0x3F],
				'=',
				'=')
		case 2:
			// 2 Bytes: xxx=
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8
			result = append(result,
				alphabet[n>>18],
				alphabet[(n>>12)&0x3F],
				alphabet[(n>>6)&0x3F],
				'=')
		default:
			// 3+ Bytes: xxxx
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8 | uint32(data[i+2])
			result = append(result,
				alphabet[n>>18],
				alphabet[(n>>12)&0x3F],
				alphabet[(n>>6)&0x3F],
				alphabet[n&0x3F])
		}
	}

	return string(result)
}
