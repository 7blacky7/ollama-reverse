//go:build mlx

// Package mlx - Mathematische Hilfsfunktionen
//
// Hauptfunktionen:
// - dotProduct: Skalarprodukt
// - magnitude: Vektorbetrag
// - cosineSimilarity: Kosinus-Ähnlichkeit
// - euclideanDistance: Euklidische Distanz
// - manhattanDistance: Manhattan-Distanz

package mlx

import "math"

// dotProduct berechnet das Skalarprodukt zweier Vektoren
func dotProduct[V float32 | float64](v1, v2 []V) V {
	var result V = 0
	if len(v1) != len(v2) {
		return result
	}
	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}
	return result
}

// magnitude berechnet die Magnitude eines Vektors
func magnitude[V float32 | float64](v []V) V {
	var result V = 0
	for _, val := range v {
		result += val * val
	}
	return V(math.Sqrt(float64(result)))
}

// cosineSimilarity berechnet die Kosinus-Ähnlichkeit
func cosineSimilarity[V float32 | float64](v1, v2 []V) V {
	mag1 := magnitude(v1)
	mag2 := magnitude(v2)
	if mag1 == 0 || mag2 == 0 {
		return 0
	}
	return dotProduct(v1, v2) / (mag1 * mag2)
}

// euclideanDistance berechnet die euklidische Distanz
func euclideanDistance[V float32 | float64](v1, v2 []V) V {
	if len(v1) != len(v2) {
		return V(math.Inf(1))
	}
	var sum V = 0
	for i := 0; i < len(v1); i++ {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return V(math.Sqrt(float64(sum)))
}

// manhattanDistance berechnet die Manhattan-Distanz
func manhattanDistance[V float32 | float64](v1, v2 []V) V {
	if len(v1) != len(v2) {
		return V(math.Inf(1))
	}
	var sum V = 0
	for i := 0; i < len(v1); i++ {
		sum += V(math.Abs(float64(v1[i] - v2[i])))
	}
	return sum
}
