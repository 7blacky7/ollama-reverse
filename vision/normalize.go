// MODUL: normalize
// ZWECK: Normalisierung und Tensor-Konvertierung fuer Vision-Modelle
// INPUT: ImageInput, Normalisierungs-Parameter (mean, std)
// OUTPUT: float32-Tensoren in verschiedenen Layouts
// NEBENEFFEKTE: keine
// ABHAENGIGKEITEN: keine (nur Standardbibliothek)
// HINWEISE: Unterstuetzt HWC und CHW Layouts, ImageNet/CLIP Presets

package vision

// Standard-Normalisierungswerte fuer verschiedene Modelle
var (
	// ImageNet Default (ResNet, EfficientNet, etc.)
	ImageNetMean = [3]float32{0.485, 0.456, 0.406}
	ImageNetStd  = [3]float32{0.229, 0.224, 0.225}

	// ImageNet Standard (normalisiert auf [-1, 1])
	ImageNetStandardMean = [3]float32{0.5, 0.5, 0.5}
	ImageNetStandardStd  = [3]float32{0.5, 0.5, 0.5}

	// CLIP Default
	ClipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	ClipStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}

	// Keine Normalisierung (nur Skalierung auf [0,1])
	NoNormMean = [3]float32{0.0, 0.0, 0.0}
	NoNormStd  = [3]float32{1.0, 1.0, 1.0}
)

// NormalizeRGB normalisiert ein Bild mit gegebenen mean/std Werten
// Gibt einen float32-Slice im CHW Format zurueck (Channel-First)
func NormalizeRGB(img *ImageInput, mean, std [3]float32) []float32 {
	bounds := img.Image.Bounds()
	h := bounds.Dy()
	w := bounds.Dx()
	size := h * w

	// Pre-allozieren fuer CHW Layout
	result := make([]float32, size*3)
	rOffset := 0
	gOffset := size
	bOffset := size * 2

	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b := extractRGB(img, x, y)

			// Normalisierung anwenden
			result[rOffset+idx] = (r - mean[0]) / std[0]
			result[gOffset+idx] = (g - mean[1]) / std[1]
			result[bOffset+idx] = (b - mean[2]) / std[2]
			idx++
		}
	}

	return result
}

// extractRGB holt RGB-Werte als float32 im Bereich [0,1]
func extractRGB(img *ImageInput, x, y int) (float32, float32, float32) {
	c := img.Image.At(x, y)
	r, g, b, _ := c.RGBA()
	// RGBA gibt 16-bit Werte zurueck, auf 8-bit konvertieren
	return float32(r>>8) / 255.0, float32(g>>8) / 255.0, float32(b>>8) / 255.0
}

// ToFloat32Tensor konvertiert ein Bild zu einem float32-Slice im HWC Format
// Werte werden auf [0,1] skaliert ohne Normalisierung
func ToFloat32Tensor(img *ImageInput) []float32 {
	bounds := img.Image.Bounds()
	h := bounds.Dy()
	w := bounds.Dx()

	// HWC Layout: Height x Width x Channels
	result := make([]float32, h*w*3)
	idx := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b := extractRGB(img, x, y)
			result[idx] = r
			result[idx+1] = g
			result[idx+2] = b
			idx += 3
		}
	}

	return result
}

// CHWTensorLayout konvertiert HWC zu CHW Layout
// Input: hwc Tensor mit Dimensionen [h, w, c]
// Output: chw Tensor mit Dimensionen [c, h, w]
func CHWTensorLayout(hwc []float32, h, w, c int) []float32 {
	if len(hwc) != h*w*c {
		return nil
	}

	chw := make([]float32, len(hwc))
	planeSize := h * w

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcIdx := (y*w + x) * c
			dstBase := y*w + x

			for ch := 0; ch < c; ch++ {
				chw[ch*planeSize+dstBase] = hwc[srcIdx+ch]
			}
		}
	}

	return chw
}

// HWCTensorLayout konvertiert CHW zu HWC Layout
// Input: chw Tensor mit Dimensionen [c, h, w]
// Output: hwc Tensor mit Dimensionen [h, w, c]
func HWCTensorLayout(chw []float32, c, h, w int) []float32 {
	if len(chw) != c*h*w {
		return nil
	}

	hwc := make([]float32, len(chw))
	planeSize := h * w

	for ch := 0; ch < c; ch++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				srcIdx := ch*planeSize + y*w + x
				dstIdx := (y*w+x)*c + ch
				hwc[dstIdx] = chw[srcIdx]
			}
		}
	}

	return hwc
}

// NormalizeRGBToHWC normalisiert und gibt HWC Layout zurueck
func NormalizeRGBToHWC(img *ImageInput, mean, std [3]float32) []float32 {
	bounds := img.Image.Bounds()
	h := bounds.Dy()
	w := bounds.Dx()

	result := make([]float32, h*w*3)
	idx := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b := extractRGB(img, x, y)

			result[idx] = (r - mean[0]) / std[0]
			result[idx+1] = (g - mean[1]) / std[1]
			result[idx+2] = (b - mean[2]) / std[2]
			idx += 3
		}
	}

	return result
}

// Dimensions gibt die Bild-Dimensionen als (H, W, C) zurueck
func (img *ImageInput) Dimensions() (int, int, int) {
	return img.Height, img.Width, 3
}

// TensorShape gibt die Tensor-Form fuer ein gegebenes Layout zurueck
func (img *ImageInput) TensorShape(channelFirst bool) []int {
	if channelFirst {
		return []int{3, img.Height, img.Width}
	}
	return []int{img.Height, img.Width, 3}
}
