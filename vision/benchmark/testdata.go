// MODUL: testdata
// ZWECK: Generierung von synthetischen Testbildern fuer Benchmarks
// INPUT: Bildgroesse (width, height), Batch-Anzahl
// OUTPUT: JPEG-kodierte Testbilder als Byte-Slices
// NEBENEFFEKTE: Keine (rein speicherbasiert)
// ABHAENGIGKEITEN: image, image/jpeg (stdlib)
// HINWEISE: Generiert Gradienten-Muster fuer realistische Kompressionsraten

package benchmark

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"math/rand"
)

// ============================================================================
// Testbild-Generierung
// ============================================================================

// GenerateTestImage generiert ein JPEG-kodiertes Testbild der angegebenen Groesse.
// Das Bild enthaelt ein Gradienten-Muster fuer realistische Kompression.
func GenerateTestImage(width, height int) []byte {
	return GenerateTestImageWithSeed(width, height, 42)
}

// GenerateTestImageWithSeed generiert ein Testbild mit definiertem Random-Seed.
// Nuetzlich fuer reproduzierbare Benchmarks.
func GenerateTestImageWithSeed(width, height int, seed int64) []byte {
	img := createGradientImage(width, height, seed)
	return encodeJPEG(img)
}

// GenerateTestBatch generiert eine Menge von Testbildern fuer Batch-Benchmarks.
func GenerateTestBatch(width, height, count int) [][]byte {
	batch := make([][]byte, count)
	for i := 0; i < count; i++ {
		// Unterschiedliche Seeds fuer Variation
		batch[i] = GenerateTestImageWithSeed(width, height, int64(i*1000))
	}
	return batch
}

// ============================================================================
// Bild-Erstellung - Muster-Generatoren
// ============================================================================

// createGradientImage erstellt ein Bild mit Farbgradient und Rauschen.
func createGradientImage(width, height int, seed int64) *image.RGBA {
	rng := rand.New(rand.NewSource(seed))
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := calculatePixelColor(x, y, width, height, rng)
			img.Set(x, y, c)
		}
	}

	return img
}

// calculatePixelColor berechnet die Farbe eines Pixels basierend auf Position.
func calculatePixelColor(x, y, width, height int, rng *rand.Rand) color.RGBA {
	// Normalisierte Koordinaten
	nx := float64(x) / float64(width)
	ny := float64(y) / float64(height)

	// Basis-Gradient
	r := uint8(nx * 255)
	g := uint8(ny * 255)
	b := uint8((nx + ny) / 2 * 255)

	// Rauschen hinzufuegen fuer Realismus
	noise := int(rng.Float64()*20 - 10)
	r = clampUint8(int(r) + noise)
	g = clampUint8(int(g) + noise)
	b = clampUint8(int(b) + noise)

	return color.RGBA{R: r, G: g, B: b, A: 255}
}

// ============================================================================
// Spezielle Testmuster
// ============================================================================

// GenerateCheckerboardImage erstellt ein Schachbrettmuster-Testbild.
// Nuetzlich fuer Edge-Case-Tests mit hohem Kontrast.
func GenerateCheckerboardImage(width, height, tileSize int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			isWhite := ((x/tileSize)+(y/tileSize))%2 == 0
			if isWhite {
				img.Set(x, y, color.White)
			} else {
				img.Set(x, y, color.Black)
			}
		}
	}

	return encodeJPEG(img)
}

// GenerateSolidColorImage erstellt ein einfarbiges Testbild.
// Nuetzlich fuer Baseline-Messungen mit minimaler Komplexitaet.
func GenerateSolidColorImage(width, height int, c color.Color) []byte {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, c)
		}
	}

	return encodeJPEG(img)
}

// ============================================================================
// Hilfsfunktionen - Encoding
// ============================================================================

// encodeJPEG kodiert ein Bild als JPEG mit Standard-Qualitaet.
func encodeJPEG(img image.Image) []byte {
	var buf bytes.Buffer
	opts := &jpeg.Options{Quality: 85}
	_ = jpeg.Encode(&buf, img, opts)
	return buf.Bytes()
}

// clampUint8 begrenzt einen int-Wert auf den uint8-Bereich.
func clampUint8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
