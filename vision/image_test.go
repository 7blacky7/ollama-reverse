// MODUL: image_test
// ZWECK: Tests fuer Bild-Lade- und Verarbeitungsfunktionen
// INPUT: Synthetische Bilder und PNG-Bytes
// OUTPUT: Testresultate
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: testing, image, image/png, bytes
// HINWEISE: Testet Resize, Crop und Composite Operationen

package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// createPNGBytes erzeugt PNG-Bytes aus einem Testbild
func createPNGBytes(w, h int, c color.Color) []byte {
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgba.Set(x, y, c)
		}
	}

	var buf bytes.Buffer
	_ = png.Encode(&buf, rgba)
	return buf.Bytes()
}

func TestLoadImageFromBytes(t *testing.T) {
	pngData := createPNGBytes(100, 50, color.RGBA{255, 0, 0, 255})

	img, err := LoadImageFromBytes(pngData)
	if err != nil {
		t.Fatalf("LoadImageFromBytes() error = %v", err)
	}

	if img.Width != 100 || img.Height != 50 {
		t.Errorf("Groesse = %dx%d, erwartet 100x50", img.Width, img.Height)
	}

	if img.Format != FormatPNG {
		t.Errorf("Format = %v, erwartet %v", img.Format, FormatPNG)
	}
}

func TestLoadImageFromBytesInvalid(t *testing.T) {
	invalidData := []byte{0x00, 0x00, 0x00, 0x00}

	_, err := LoadImageFromBytes(invalidData)
	if err == nil {
		t.Error("Erwartet Fehler bei ungueltigem Format")
	}
}

func TestDecodeImage(t *testing.T) {
	pngData := createPNGBytes(80, 60, color.White)
	reader := bytes.NewReader(pngData)

	img, err := DecodeImage(reader)
	if err != nil {
		t.Fatalf("DecodeImage() error = %v", err)
	}

	if img.Width != 80 || img.Height != 60 {
		t.Errorf("Groesse = %dx%d, erwartet 80x60", img.Width, img.Height)
	}
}

func TestResizeImage(t *testing.T) {
	pngData := createPNGBytes(100, 100, color.White)
	img, _ := LoadImageFromBytes(pngData)

	resized, err := ResizeImage(img, 50, 50)
	if err != nil {
		t.Fatalf("ResizeImage() error = %v", err)
	}

	if resized.Width != 50 || resized.Height != 50 {
		t.Errorf("Groesse = %dx%d, erwartet 50x50", resized.Width, resized.Height)
	}
}

func TestResizeImageInvalidSize(t *testing.T) {
	pngData := createPNGBytes(100, 100, color.White)
	img, _ := LoadImageFromBytes(pngData)

	_, err := ResizeImage(img, 0, 50)
	if err == nil {
		t.Error("Erwartet Fehler bei Breite 0")
	}

	_, err = ResizeImage(img, 50, -1)
	if err == nil {
		t.Error("Erwartet Fehler bei negativer Hoehe")
	}
}

func TestResizeWithAspect(t *testing.T) {
	// 200x100 Bild (2:1 Seitenverhaeltnis)
	pngData := createPNGBytes(200, 100, color.White)
	img, _ := LoadImageFromBytes(pngData)

	// Max 100x100 sollte 100x50 ergeben
	resized, err := ResizeWithAspect(img, 100, 100)
	if err != nil {
		t.Fatalf("ResizeWithAspect() error = %v", err)
	}

	if resized.Width != 100 || resized.Height != 50 {
		t.Errorf("Groesse = %dx%d, erwartet 100x50", resized.Width, resized.Height)
	}
}

func TestComposite(t *testing.T) {
	// Transparentes Bild
	rgba := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			rgba.Set(x, y, color.RGBA{255, 0, 0, 128}) // Halbtransparentes Rot
		}
	}

	img := &ImageInput{Image: rgba, Width: 10, Height: 10, Format: FormatPNG}
	composited := Composite(img)

	// Nach Composite sollte Alpha 255 sein
	r, g, b, a := composited.Image.At(5, 5).RGBA()
	if a>>8 != 255 {
		t.Errorf("Alpha = %d, erwartet 255", a>>8)
	}

	// Farbe sollte gemischt sein (rot + weiss)
	if r>>8 < 127 || r>>8 > 255 {
		t.Errorf("Rot = %d, erwartet zwischen 127 und 255", r>>8)
	}
	_ = g
	_ = b
}

func TestCenterCrop(t *testing.T) {
	pngData := createPNGBytes(100, 100, color.White)
	img, _ := LoadImageFromBytes(pngData)

	cropped, err := CenterCrop(img, 50, 50)
	if err != nil {
		t.Fatalf("CenterCrop() error = %v", err)
	}

	if cropped.Width != 50 || cropped.Height != 50 {
		t.Errorf("Groesse = %dx%d, erwartet 50x50", cropped.Width, cropped.Height)
	}
}

func TestCenterCropTooLarge(t *testing.T) {
	pngData := createPNGBytes(50, 50, color.White)
	img, _ := LoadImageFromBytes(pngData)

	_, err := CenterCrop(img, 100, 100)
	if err == nil {
		t.Error("Erwartet Fehler wenn Crop groesser als Bild")
	}
}

func TestCalculateAspectSize(t *testing.T) {
	tests := []struct {
		srcW, srcH, maxW, maxH int
		expectW, expectH       int
	}{
		{200, 100, 100, 100, 100, 50},  // Breites Bild
		{100, 200, 100, 100, 50, 100},  // Hohes Bild
		{100, 100, 200, 200, 200, 200}, // Kleineres Bild hochskalieren
		{50, 50, 100, 50, 50, 50},      // Hoehe begrenzt
	}

	for _, tt := range tests {
		w, h := calculateAspectSize(tt.srcW, tt.srcH, tt.maxW, tt.maxH)
		if w != tt.expectW || h != tt.expectH {
			t.Errorf("calculateAspectSize(%d,%d,%d,%d) = (%d,%d), erwartet (%d,%d)",
				tt.srcW, tt.srcH, tt.maxW, tt.maxH, w, h, tt.expectW, tt.expectH)
		}
	}
}
