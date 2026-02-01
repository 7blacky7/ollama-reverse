// MODUL: image
// ZWECK: Bild-Lade- und Verarbeitungsfunktionen fuer Vision-Modelle
// INPUT: Dateipfad, Bytes oder io.Reader
// OUTPUT: ImageInput Struktur mit dekodiertem Bild
// NEBENEFFEKTE: Dateisystem-Lesezugriff bei LoadImage
// ABHAENGIGKEITEN: golang.org/x/image/draw (extern), image/jpeg, image/png
// HINWEISE: Alle Bilder werden als RGBA konvertiert, WebP benoetigt x/image/webp

package vision

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"

	// Standard-Decoder registrieren
	_ "image/jpeg"
	_ "image/png"

	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// ImageInput enthaelt ein dekodiertes Bild mit Metadaten
type ImageInput struct {
	Image  *image.RGBA
	Width  int
	Height int
	Format ImageFormat
}

// LoadImage laedt ein Bild von einem Dateipfad
func LoadImage(path string) (*ImageInput, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("datei lesen fehlgeschlagen: %w", err)
	}
	return LoadImageFromBytes(data)
}

// LoadImageFromBytes dekodiert ein Bild aus Byte-Daten
func LoadImageFromBytes(data []byte) (*ImageInput, error) {
	format := DetectFormat(data)
	if err := ValidateFormat(format); err != nil {
		return nil, err
	}

	reader := bytes.NewReader(data)
	return decodeWithFormat(reader, format)
}

// DecodeImage dekodiert ein Bild aus einem io.Reader
func DecodeImage(reader io.Reader) (*ImageInput, error) {
	// Erst Daten puffern fuer Format-Erkennung
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("daten lesen fehlgeschlagen: %w", err)
	}
	return LoadImageFromBytes(data)
}

// decodeWithFormat dekodiert und konvertiert zu RGBA
func decodeWithFormat(reader io.Reader, format ImageFormat) (*ImageInput, error) {
	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("bild dekodieren fehlgeschlagen: %w", err)
	}

	rgba := toRGBA(img)
	bounds := rgba.Bounds()

	return &ImageInput{
		Image:  rgba,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Format: format,
	}, nil
}

// toRGBA konvertiert ein beliebiges image.Image zu *image.RGBA
func toRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	return rgba
}

// ResizeImage skaliert ein Bild auf die angegebene Groesse
func ResizeImage(img *ImageInput, width, height int) (*ImageInput, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("ungueltige Groesse: %dx%d", width, height)
	}

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(dst, dst.Bounds(), img.Image, img.Image.Bounds(), draw.Over, nil)

	return &ImageInput{
		Image:  dst,
		Width:  width,
		Height: height,
		Format: img.Format,
	}, nil
}

// ResizeWithAspect skaliert unter Beibehaltung des Seitenverhaeltnisses
func ResizeWithAspect(img *ImageInput, maxWidth, maxHeight int) (*ImageInput, error) {
	if maxWidth <= 0 || maxHeight <= 0 {
		return nil, fmt.Errorf("ungueltige Groesse: %dx%d", maxWidth, maxHeight)
	}

	newW, newH := calculateAspectSize(img.Width, img.Height, maxWidth, maxHeight)
	return ResizeImage(img, newW, newH)
}

// calculateAspectSize berechnet Zielgroesse mit Seitenverhaeltnis
func calculateAspectSize(srcW, srcH, maxW, maxH int) (int, int) {
	ratioW := float64(maxW) / float64(srcW)
	ratioH := float64(maxH) / float64(srcH)

	ratio := ratioW
	if ratioH < ratioW {
		ratio = ratioH
	}

	return int(float64(srcW) * ratio), int(float64(srcH) * ratio)
}

// Composite entfernt Alpha-Kanal durch weissen Hintergrund
func Composite(img *ImageInput) *ImageInput {
	return CompositeWithColor(img, color.White)
}

// CompositeWithColor entfernt Alpha-Kanal mit gegebener Hintergrundfarbe
func CompositeWithColor(img *ImageInput, bgColor color.Color) *ImageInput {
	bounds := img.Image.Bounds()
	dst := image.NewRGBA(bounds)

	// Hintergrund fuellen
	draw.Draw(dst, bounds, &image.Uniform{bgColor}, image.Point{}, draw.Src)
	// Bild darueber zeichnen
	draw.Draw(dst, bounds, img.Image, bounds.Min, draw.Over)

	return &ImageInput{
		Image:  dst,
		Width:  img.Width,
		Height: img.Height,
		Format: img.Format,
	}
}

// CenterCrop schneidet einen zentrierten Bereich aus
func CenterCrop(img *ImageInput, width, height int) (*ImageInput, error) {
	if width > img.Width || height > img.Height {
		return nil, fmt.Errorf("crop groesser als bild: %dx%d > %dx%d", width, height, img.Width, img.Height)
	}

	offsetX := (img.Width - width) / 2
	offsetY := (img.Height - height) / 2

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	srcRect := image.Rect(offsetX, offsetY, offsetX+width, offsetY+height)

	draw.Draw(dst, dst.Bounds(), img.Image, srcRect.Min, draw.Src)

	return &ImageInput{
		Image:  dst,
		Width:  width,
		Height: height,
		Format: img.Format,
	}, nil
}
