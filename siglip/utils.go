package siglip

// ============================================================================
// Image
// ============================================================================

// Image repraesentiert ein Bild fuer SigLIP
type Image struct {
	Data     []byte
	Width    int
	Height   int
	Channels int
}

// NewImageFromRGB erstellt ein neues Image aus RGB-Daten
func NewImageFromRGB(data []byte, width, height int) *Image {
	return &Image{
		Data:     data,
		Width:    width,
		Height:   height,
		Channels: 3,
	}
}

// ============================================================================
// Similarity Functions
// ============================================================================

// CosineSimilarityMatrix berechnet die Cosine Similarity Matrix fuer mehrere Embeddings
func CosineSimilarityMatrix(embeddings []*Embedding) [][]float32 {
	n := len(embeddings)
	matrix := make([][]float32, n)

	for i := range matrix {
		matrix[i] = make([]float32, n)
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0
			} else if i < j {
				matrix[i][j] = embeddings[i].CosineSimilarity(embeddings[j])
			} else {
				matrix[i][j] = matrix[j][i] // Symmetrie nutzen
			}
		}
	}

	return matrix
}

// FindMostSimilar findet die aehnlichsten Embeddings zu einem Query-Embedding
func FindMostSimilar(query *Embedding, candidates []*Embedding, topK int) []int {
	if query == nil || len(candidates) == 0 {
		return nil
	}

	// Similarities berechnen
	type scored struct {
		index int
		score float32
	}

	scores := make([]scored, len(candidates))
	for i, cand := range candidates {
		scores[i] = scored{
			index: i,
			score: query.CosineSimilarity(cand),
		}
	}

	// Sortieren (einfaches Bubble Sort fuer kleine Listen)
	for i := 0; i < len(scores)-1; i++ {
		for j := 0; j < len(scores)-i-1; j++ {
			if scores[j].score < scores[j+1].score {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	// Top-K extrahieren
	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = scores[i].index
	}

	return result
}

// ============================================================================
// Helper Functions
// ============================================================================

// sqrt64 ist eine Hilfsfunktion fuer Quadratwurzel
func sqrt64(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	// Newton-Raphson
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// base64Encode kodiert Bytes zu Base64
func base64Encode(data []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

	result := make([]byte, 0, (len(data)+2)/3*4)

	for i := 0; i < len(data); i += 3 {
		var n uint32

		remaining := len(data) - i
		switch remaining {
		case 1:
			n = uint32(data[i]) << 16
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], '=', '=')
		case 2:
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], alphabet[(n>>6)&0x3F], '=')
		default:
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8 | uint32(data[i+2])
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], alphabet[(n>>6)&0x3F], alphabet[n&0x3F])
		}
	}

	return string(result)
}
