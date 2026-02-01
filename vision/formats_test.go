// MODUL: formats_test
// ZWECK: Tests fuer Format-Erkennung und Validierung
// INPUT: Test-Bytes mit verschiedenen Signaturen
// OUTPUT: Testresultate
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: testing
// HINWEISE: Testet Magic-Byte-Erkennung fuer JPEG/PNG/WebP

package vision

import (
	"testing"
)

func TestDetectFormat(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected ImageFormat
	}{
		{
			name:     "JPEG Magic Bytes",
			data:     []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10},
			expected: FormatJPEG,
		},
		{
			name:     "PNG Magic Bytes",
			data:     []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A},
			expected: FormatPNG,
		},
		{
			name:     "WebP Magic Bytes",
			data:     []byte{0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 'W', 'E', 'B', 'P'},
			expected: FormatWebP,
		},
		{
			name:     "Zu kurze Daten",
			data:     []byte{0xFF, 0xD8},
			expected: FormatUnknown,
		},
		{
			name:     "Unbekanntes Format",
			data:     []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
			expected: FormatUnknown,
		},
		{
			name:     "Leere Daten",
			data:     []byte{},
			expected: FormatUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DetectFormat(tt.data)
			if result != tt.expected {
				t.Errorf("DetectFormat() = %v, erwartet %v", result, tt.expected)
			}
		})
	}
}

func TestValidateFormat(t *testing.T) {
	tests := []struct {
		format    ImageFormat
		expectErr bool
	}{
		{FormatJPEG, false},
		{FormatPNG, false},
		{FormatWebP, false},
		{FormatUnknown, true},
		{ImageFormat("gif"), true},
	}

	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			err := ValidateFormat(tt.format)
			if (err != nil) != tt.expectErr {
				t.Errorf("ValidateFormat(%v) error = %v, expectErr %v", tt.format, err, tt.expectErr)
			}
		})
	}
}

func TestImageFormatMimeType(t *testing.T) {
	tests := []struct {
		format   ImageFormat
		expected string
	}{
		{FormatJPEG, "image/jpeg"},
		{FormatPNG, "image/png"},
		{FormatWebP, "image/webp"},
		{FormatUnknown, "application/octet-stream"},
	}

	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			if got := tt.format.MimeType(); got != tt.expected {
				t.Errorf("MimeType() = %v, erwartet %v", got, tt.expected)
			}
		})
	}
}

func TestImageFormatExtension(t *testing.T) {
	tests := []struct {
		format   ImageFormat
		expected string
	}{
		{FormatJPEG, ".jpg"},
		{FormatPNG, ".png"},
		{FormatWebP, ".webp"},
		{FormatUnknown, ".bin"},
	}

	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			if got := tt.format.Extension(); got != tt.expected {
				t.Errorf("Extension() = %v, erwartet %v", got, tt.expected)
			}
		})
	}
}
