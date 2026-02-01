//go:build mlx

// image_utils.go - Bild-Hilfsfunktionen fuer FLUX.2.
//
// Dieses Modul enthaelt:
// - PrepareImage fuer Resize und Cropping
// - ImageToTensor Konvertierung
// - EncodeImageRefs fuer Referenzbild-Encoding

package flux2

import (
	"fmt"
	"image"
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"golang.org/x/image/draw"
)

// PrepareImage resizes and crops an image to be a multiple of 16, with optional pixel limit.
// Returns the processed image and its dimensions.
func PrepareImage(img image.Image, limitPixels int) (image.Image, int, int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Cap pixels if needed (like diffusers cap_pixels)
	if limitPixels > 0 && w*h > limitPixels {
		scale := math.Sqrt(float64(limitPixels) / float64(w*h))
		w = int(float64(w) * scale)
		h = int(float64(h) * scale)
	}

	// Round down to multiple of 16
	w = (w / 16) * 16
	h = (h / 16) * 16

	if w < 16 {
		w = 16
	}
	if h < 16 {
		h = 16
	}

	// Resize using high-quality bicubic interpolation (matches diffusers' default lanczos)
	resized := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	return resized, w, h
}

// ImageToTensor converts an image to a tensor in [-1, 1] range with shape [1, C, H, W].
func ImageToTensor(img image.Image) *mlx.Array {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Convert to float32 array in NCHW format [1, 3, H, W] with values in [-1, 1]
	data := make([]float32, 3*h*w)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			// RGBA returns 16-bit values, convert to [-1, 1]
			data[0*h*w+y*w+x] = float32(r>>8)/127.5 - 1.0
			data[1*h*w+y*w+x] = float32(g>>8)/127.5 - 1.0
			data[2*h*w+y*w+x] = float32(b>>8)/127.5 - 1.0
		}
	}

	arr := mlx.NewArrayFloat32(data, []int32{1, 3, int32(h), int32(w)})
	return arr
}

// EncodeImageRefs encodes reference images using the VAE.
func (m *Model) EncodeImageRefs(images []image.Image) (*ImageCondTokens, error) {
	if len(images) == 0 {
		return nil, nil
	}

	// Limit reference images to reduce attention memory
	limitPixels := MaxRefPixels
	if len(images) > 1 {
		limitPixels = MaxRefPixels / 2
	}

	var allTokens []*mlx.Array

	for _, img := range images {
		// Prepare image (resize, crop to multiple of 16)
		prepared, prepW, prepH := PrepareImage(img, limitPixels)
		fmt.Printf("    Encoding %dx%d image... ", prepW, prepH)

		// Convert to tensor [-1, 1]
		tensor := ImageToTensor(prepared)

		// Encode with VAE - returns [1, L, 128]
		encoded := m.VAE.EncodeImage(tensor)
		squeezed := mlx.Squeeze(encoded, 0) // [L, C]

		// Defer eval - will be done with other setup arrays
		allTokens = append(allTokens, squeezed)
		fmt.Println("OK")
	}

	// For single image, just add batch dimension directly
	// For multiple images, concatenate first
	var tokens *mlx.Array
	if len(allTokens) == 1 {
		tokens = mlx.ExpandDims(allTokens[0], 0) // [1, L, C]
	} else {
		tokens = mlx.Concatenate(allTokens, 0) // [total_L, C]
		tokens = mlx.ExpandDims(tokens, 0)     // [1, total_L, C]
	}

	return &ImageCondTokens{Tokens: tokens}, nil
}
