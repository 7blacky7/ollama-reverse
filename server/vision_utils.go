// MODUL: vision_utils
// ZWECK: Utility-Funktionen fuer die Vision API (Base64, Vektoroperationen, Validierung)
// INPUT: Base64-Strings, float32-Vektoren, Bild-Bytes
// OUTPUT: Dekodierte Bytes, Similarity-Scores, normalisierte Vektoren
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: encoding/base64, math (Standardbibliothek)
// HINWEISE: Alle Vektor-Funktionen erwarten gleich lange Vektoren
package server

import (
	"encoding/base64"
	"errors"
	"math"
)

// ============================================================================
// Fehler fuer Utility-Funktionen
// ============================================================================

var (
	// ErrVectorLengthMismatch wird geworfen wenn Vektoren unterschiedliche Laenge haben
	ErrVectorLengthMismatch = errors.New("vector length mismatch")

	// ErrEmptyVector wird geworfen bei leerem Vektor
	ErrEmptyVector = errors.New("empty vector")

	// ErrZeroMagnitude wird geworfen wenn Vektor-Betrag null ist
	ErrZeroMagnitude = errors.New("zero magnitude vector")
)

// ============================================================================
// Base64 Funktionen
// ============================================================================

// decodeBase64Image dekodiert einen Base64-String zu Bytes.
// Unterstuetzt Standard und URL-safe Base64.
func decodeBase64Image(b64 string) ([]byte, error) {
	// Standard-Encoding versuchen
	data, err := base64.StdEncoding.DecodeString(b64)
	if err == nil {
		return data, nil
	}

	// URL-safe Encoding als Fallback
	data, err = base64.URLEncoding.DecodeString(b64)
	if err != nil {
		return nil, ErrVisionInvalidBase64
	}

	return data, nil
}

// ============================================================================
// Vektor-Operationen
// ============================================================================

// NOTE: cosineSimilarity is now in handlers_vision_similarity.go to avoid duplication

// dotProduct berechnet das Skalarprodukt zweier Vektoren.
func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// magnitude berechnet den Betrag (L2-Norm) eines Vektors.
func magnitude(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}

	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}

	return float32(math.Sqrt(sum))
}

// normalizeVector fuehrt L2-Normalisierung auf einem Vektor durch.
// Gibt einen neuen Vektor zurueck ohne den Original zu veraendern.
func normalizeVector(v []float32) []float32 {
	if len(v) == 0 {
		return nil
	}

	mag := magnitude(v)
	if mag == 0 {
		// Null-Vektor kann nicht normalisiert werden
		return make([]float32, len(v))
	}

	normalized := make([]float32, len(v))
	for i, val := range v {
		normalized[i] = val / mag
	}

	return normalized
}

// ============================================================================
// Bild-Validierung
// ============================================================================

// Unterstuetzte Bild-Magic-Bytes
var (
	jpegMagic = []byte{0xFF, 0xD8, 0xFF}
	pngMagic  = []byte{0x89, 0x50, 0x4E, 0x47}
	gifMagic  = []byte{0x47, 0x49, 0x46}
	webpMagic = []byte{0x52, 0x49, 0x46, 0x46}
	bmpMagic  = []byte{0x42, 0x4D}
)

// validateImageFormat prueft ob die Bild-Daten ein gueltiges Format haben.
// Unterstuetzt JPEG, PNG, GIF, WebP und BMP.
func validateImageFormat(data []byte) error {
	if len(data) < 4 {
		return ErrVisionInvalidImage
	}

	// JPEG pruefen
	if hasPrefix(data, jpegMagic) {
		return nil
	}

	// PNG pruefen
	if hasPrefix(data, pngMagic) {
		return nil
	}

	// GIF pruefen
	if hasPrefix(data, gifMagic) {
		return nil
	}

	// WebP pruefen (RIFF....WEBP)
	if hasPrefix(data, webpMagic) && len(data) >= 12 {
		if data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P' {
			return nil
		}
	}

	// BMP pruefen
	if hasPrefix(data, bmpMagic) {
		return nil
	}

	return ErrVisionInvalidImage
}

// hasPrefix prueft ob data mit prefix beginnt.
func hasPrefix(data, prefix []byte) bool {
	if len(data) < len(prefix) {
		return false
	}

	for i, b := range prefix {
		if data[i] != b {
			return false
		}
	}

	return true
}
