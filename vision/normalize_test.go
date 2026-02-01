// MODUL: normalize_test
// ZWECK: Tests fuer Normalisierungs- und Tensor-Funktionen
// INPUT: Synthetische Bilder
// OUTPUT: Testresultate
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: testing, image
// HINWEISE: Testet CHW/HWC Konvertierung und Normalisierungswerte

package vision

import (
	"image"
	"image/color"
	"testing"
)

// createTestImage erzeugt ein einfaches Testbild
func createTestImage(w, h int, c color.Color) *ImageInput {
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgba.Set(x, y, c)
		}
	}
	return &ImageInput{
		Image:  rgba,
		Width:  w,
		Height: h,
		Format: FormatPNG,
	}
}

func TestToFloat32Tensor(t *testing.T) {
	// Rotes 2x2 Bild
	img := createTestImage(2, 2, color.RGBA{255, 0, 0, 255})
	tensor := ToFloat32Tensor(img)

	// Erwarte 2*2*3 = 12 Werte im HWC Format
	expectedLen := 12
	if len(tensor) != expectedLen {
		t.Errorf("Tensor Laenge = %d, erwartet %d", len(tensor), expectedLen)
	}

	// Erstes Pixel sollte [1.0, 0.0, 0.0] sein
	if tensor[0] != 1.0 {
		t.Errorf("R-Kanal = %f, erwartet 1.0", tensor[0])
	}
	if tensor[1] != 0.0 {
		t.Errorf("G-Kanal = %f, erwartet 0.0", tensor[1])
	}
	if tensor[2] != 0.0 {
		t.Errorf("B-Kanal = %f, erwartet 0.0", tensor[2])
	}
}

func TestNormalizeRGB(t *testing.T) {
	// Graues Bild (127, 127, 127) ~ 0.5 nach Skalierung
	img := createTestImage(2, 2, color.RGBA{127, 127, 127, 255})

	// Standard-Normalisierung: (0.5 - 0.5) / 0.5 = 0
	result := NormalizeRGB(img, ImageNetStandardMean, ImageNetStandardStd)

	// CHW Format: 3 Channels mit je 4 Werten
	expectedLen := 12
	if len(result) != expectedLen {
		t.Errorf("Tensor Laenge = %d, erwartet %d", len(result), expectedLen)
	}

	// Bei 127/255 ~ 0.498, (0.498 - 0.5) / 0.5 ~ -0.004
	tolerance := float32(0.01)
	if result[0] > tolerance || result[0] < -tolerance {
		t.Errorf("Normalisierter Wert = %f, erwartet ~0", result[0])
	}
}

func TestCHWTensorLayout(t *testing.T) {
	// HWC: [R0, G0, B0, R1, G1, B1, R2, G2, B2, R3, G3, B3] fuer 2x2 Bild
	hwc := []float32{
		1, 2, 3, // Pixel 0,0
		4, 5, 6, // Pixel 1,0
		7, 8, 9, // Pixel 0,1
		10, 11, 12, // Pixel 1,1
	}

	chw := CHWTensorLayout(hwc, 2, 2, 3)

	// CHW: [R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3]
	expectedR := []float32{1, 4, 7, 10}
	expectedG := []float32{2, 5, 8, 11}
	expectedB := []float32{3, 6, 9, 12}

	// Pruefe R-Kanal
	for i, v := range expectedR {
		if chw[i] != v {
			t.Errorf("R-Kanal[%d] = %f, erwartet %f", i, chw[i], v)
		}
	}
	// Pruefe G-Kanal
	for i, v := range expectedG {
		if chw[4+i] != v {
			t.Errorf("G-Kanal[%d] = %f, erwartet %f", i, chw[4+i], v)
		}
	}
	// Pruefe B-Kanal
	for i, v := range expectedB {
		if chw[8+i] != v {
			t.Errorf("B-Kanal[%d] = %f, erwartet %f", i, chw[8+i], v)
		}
	}
}

func TestHWCTensorLayout(t *testing.T) {
	// CHW: [R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3]
	chw := []float32{
		1, 4, 7, 10, // R-Kanal
		2, 5, 8, 11, // G-Kanal
		3, 6, 9, 12, // B-Kanal
	}

	hwc := HWCTensorLayout(chw, 3, 2, 2)

	// HWC: [R0, G0, B0, R1, G1, B1, ...]
	expected := []float32{
		1, 2, 3, // Pixel 0,0
		4, 5, 6, // Pixel 1,0
		7, 8, 9, // Pixel 0,1
		10, 11, 12, // Pixel 1,1
	}

	for i, v := range expected {
		if hwc[i] != v {
			t.Errorf("HWC[%d] = %f, erwartet %f", i, hwc[i], v)
		}
	}
}

func TestImageInputDimensions(t *testing.T) {
	img := createTestImage(100, 50, color.White)

	h, w, c := img.Dimensions()
	if h != 50 || w != 100 || c != 3 {
		t.Errorf("Dimensions() = (%d, %d, %d), erwartet (50, 100, 3)", h, w, c)
	}
}

func TestImageInputTensorShape(t *testing.T) {
	img := createTestImage(100, 50, color.White)

	// CHW Layout
	chw := img.TensorShape(true)
	if chw[0] != 3 || chw[1] != 50 || chw[2] != 100 {
		t.Errorf("TensorShape(true) = %v, erwartet [3, 50, 100]", chw)
	}

	// HWC Layout
	hwc := img.TensorShape(false)
	if hwc[0] != 50 || hwc[1] != 100 || hwc[2] != 3 {
		t.Errorf("TensorShape(false) = %v, erwartet [50, 100, 3]", hwc)
	}
}

func TestCHWHWCRoundtrip(t *testing.T) {
	// Originales HWC
	hwc := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	// Hin- und Rueckkonvertierung
	chw := CHWTensorLayout(hwc, 2, 2, 3)
	result := HWCTensorLayout(chw, 3, 2, 2)

	for i, v := range hwc {
		if result[i] != v {
			t.Errorf("Roundtrip[%d] = %f, erwartet %f", i, result[i], v)
		}
	}
}
