// MODUL: formats
// ZWECK: Bildformat-Erkennung und Validierung fuer Vision-Modelle
// INPUT: Bild-Bytes oder Format-String
// OUTPUT: ImageFormat, Fehler bei ungueltigem Format
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: keine (nur Standardbibliothek)
// HINWEISE: Magic-Bytes-basierte Erkennung, unterstuetzt JPEG/PNG/WebP

package vision

import (
	"errors"
)

// ImageFormat repraesentiert ein unterstuetztes Bildformat
type ImageFormat string

const (
	FormatJPEG    ImageFormat = "jpeg"
	FormatPNG     ImageFormat = "png"
	FormatWebP    ImageFormat = "webp"
	FormatUnknown ImageFormat = "unknown"
)

// Magic-Byte-Signaturen fuer Bildformate
var (
	magicJPEG = []byte{0xFF, 0xD8, 0xFF}
	magicPNG  = []byte{0x89, 0x50, 0x4E, 0x47}
	magicWebP = []byte{0x52, 0x49, 0x46, 0x46} // "RIFF" header
)

// ErrUnknownFormat wird zurueckgegeben wenn Format nicht erkannt wurde
var ErrUnknownFormat = errors.New("unbekanntes Bildformat")

// ErrUnsupportedFormat wird zurueckgegeben bei ungueltigem Format
var ErrUnsupportedFormat = errors.New("nicht unterstuetztes Bildformat")

// DetectFormat erkennt das Bildformat anhand der Magic-Bytes
func DetectFormat(data []byte) ImageFormat {
	if len(data) < 4 {
		return FormatUnknown
	}

	if matchesMagic(data, magicJPEG) {
		return FormatJPEG
	}

	if matchesMagic(data, magicPNG) {
		return FormatPNG
	}

	if matchesMagic(data, magicWebP) && isValidWebP(data) {
		return FormatWebP
	}

	return FormatUnknown
}

// matchesMagic prueft ob die Daten mit der Signatur beginnen
func matchesMagic(data, magic []byte) bool {
	if len(data) < len(magic) {
		return false
	}
	for i, b := range magic {
		if data[i] != b {
			return false
		}
	}
	return true
}

// isValidWebP prueft auf "WEBP" Marker nach RIFF Header
func isValidWebP(data []byte) bool {
	if len(data) < 12 {
		return false
	}
	// RIFF....WEBP
	return data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P'
}

// ValidateFormat prueft ob ein Format unterstuetzt wird
func ValidateFormat(format ImageFormat) error {
	switch format {
	case FormatJPEG, FormatPNG, FormatWebP:
		return nil
	case FormatUnknown:
		return ErrUnknownFormat
	default:
		return ErrUnsupportedFormat
	}
}

// MimeType gibt den MIME-Type fuer ein Format zurueck
func (f ImageFormat) MimeType() string {
	switch f {
	case FormatJPEG:
		return "image/jpeg"
	case FormatPNG:
		return "image/png"
	case FormatWebP:
		return "image/webp"
	default:
		return "application/octet-stream"
	}
}

// Extension gibt die Dateiendung fuer ein Format zurueck
func (f ImageFormat) Extension() string {
	switch f {
	case FormatJPEG:
		return ".jpg"
	case FormatPNG:
		return ".png"
	case FormatWebP:
		return ".webp"
	default:
		return ".bin"
	}
}

// String implementiert Stringer Interface
func (f ImageFormat) String() string {
	return string(f)
}
