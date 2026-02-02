//go:build vision && cgo

// MODUL: onnx/preprocess
// ZWECK: Bild-Preprocessing fuer ONNX Vision Modelle
// INPUT: Bild-Pfad oder Bytes, Ziel-Groesse
// OUTPUT: float32 Tensor im NCHW Format [1, 3, H, W]
// NEBENEFFEKTE: Dateisystem-Lesezugriff bei LoadImage
// ABHAENGIGKEITEN: vision (ImageInput, ResizeImage)
// HINWEISE: Verwendet ImageNet Normalisierung (mean/std)

package onnx

import (
	"fmt"
	"image"

	"github.com/ollama/ollama/vision"
)

// ============================================================================
// Normalisierungs-Konstanten
// ============================================================================

// ImageNet Normalisierungswerte (Standard fuer Nomic Embed Vision)
var (
	ImageNetMean = [3]float32{0.485, 0.456, 0.406}
	ImageNetStd  = [3]float32{0.229, 0.224, 0.225}
)

// ============================================================================
// Bild laden
// ============================================================================

// LoadImage laedt ein Bild von einem Dateipfad.
// Unterstuetzt JPEG, PNG, WebP Formate.
func LoadImage(path string) (image.Image, error) {
	imgInput, err := vision.LoadImage(path)
	if err != nil {
		return nil, fmt.Errorf("bild laden fehlgeschlagen: %w", err)
	}
	return imgInput.Image, nil
}

// LoadImageFromBytes dekodiert ein Bild aus Byte-Daten.
func LoadImageFromBytes(data []byte) (image.Image, error) {
	imgInput, err := vision.LoadImageFromBytes(data)
	if err != nil {
		return nil, fmt.Errorf("bild dekodieren fehlgeschlagen: %w", err)
	}
	return imgInput.Image, nil
}

// ============================================================================
// Preprocessing Pipeline
// ============================================================================

// PreprocessImage fuehrt das vollstaendige Preprocessing durch.
// 1. Resize auf targetSize x targetSize
// 2. Konvertierung zu float32
// 3. Normalisierung mit ImageNet mean/std
// 4. Konvertierung zu NCHW Format
//
// Rueckgabe: float32 Slice der Laenge 3 * targetSize * targetSize
func PreprocessImage(img image.Image, targetSize int) []float32 {
	// Zu ImageInput konvertieren fuer Resize
	imgInput := toImageInput(img)

	// Auf Zielgroesse skalieren
	resized, err := vision.ResizeImage(imgInput, targetSize, targetSize)
	if err != nil {
		// Fallback: Original verwenden
		resized = imgInput
	}

	// NCHW Tensor erstellen
	return imageToNCHW(resized.Image, targetSize, ImageNetMean, ImageNetStd)
}

// PreprocessFromBytes laedt und preprocessed Bild aus Bytes.
func PreprocessFromBytes(data []byte, targetSize int) ([]float32, error) {
	imgInput, err := vision.LoadImageFromBytes(data)
	if err != nil {
		return nil, err
	}

	// Resize
	resized, err := vision.ResizeImage(imgInput, targetSize, targetSize)
	if err != nil {
		return nil, fmt.Errorf("resize fehlgeschlagen: %w", err)
	}

	return imageToNCHW(resized.Image, targetSize, ImageNetMean, ImageNetStd), nil
}

// PreprocessFromPath laedt und preprocessed Bild von Pfad.
func PreprocessFromPath(path string, targetSize int) ([]float32, error) {
	imgInput, err := vision.LoadImage(path)
	if err != nil {
		return nil, err
	}

	resized, err := vision.ResizeImage(imgInput, targetSize, targetSize)
	if err != nil {
		return nil, fmt.Errorf("resize fehlgeschlagen: %w", err)
	}

	return imageToNCHW(resized.Image, targetSize, ImageNetMean, ImageNetStd), nil
}

// ============================================================================
// Tensor-Konvertierung
// ============================================================================

// imageToNCHW konvertiert ein Bild zu einem NCHW float32 Tensor.
// Format: [Channels, Height, Width] - Channel-First fuer ONNX
func imageToNCHW(img *image.RGBA, size int, mean, std [3]float32) []float32 {
	tensor := make([]float32, 3*size*size)
	planeSize := size * size

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			// RGBA uint32 (0-65535) zu float32 (0-1)
			rf := float32(r) / 65535.0
			gf := float32(g) / 65535.0
			bf := float32(b) / 65535.0

			// Normalisieren und in NCHW Format speichern
			idx := y*size + x
			tensor[0*planeSize+idx] = (rf - mean[0]) / std[0] // R
			tensor[1*planeSize+idx] = (gf - mean[1]) / std[1] // G
			tensor[2*planeSize+idx] = (bf - mean[2]) / std[2] // B
		}
	}

	return tensor
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// toImageInput konvertiert ein image.Image zu vision.ImageInput
func toImageInput(img image.Image) *vision.ImageInput {
	bounds := img.Bounds()

	// Falls bereits RGBA, direkt verwenden
	if rgba, ok := img.(*image.RGBA); ok {
		return &vision.ImageInput{
			Image:  rgba,
			Width:  bounds.Dx(),
			Height: bounds.Dy(),
		}
	}

	// Sonst konvertieren
	rgba := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	return &vision.ImageInput{
		Image:  rgba,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
	}
}
