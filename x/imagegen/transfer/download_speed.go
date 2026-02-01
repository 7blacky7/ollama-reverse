// Package transfer - Geschwindigkeitstracking für Downloads
//
// Diese Datei enthält den speedTracker für adaptive Download-Erkennung.
// Der Tracker speichert die letzten Download-Geschwindigkeiten und berechnet
// den Median, um langsame Downloads zu erkennen.
//
// Hauptfunktionen:
// - record: Zeichnet eine Geschwindigkeit auf
// - median: Berechnet den Median der letzten Geschwindigkeiten
package transfer

import (
	"slices"
	"sync"
)

// speedTracker verfolgt Download-Geschwindigkeiten für adaptive Erkennung
type speedTracker struct {
	mu     sync.Mutex
	speeds []float64
}

// record zeichnet eine Download-Geschwindigkeit auf (Bytes/Sekunde)
func (s *speedTracker) record(v float64) {
	s.mu.Lock()
	s.speeds = append(s.speeds, v)
	// Maximal 30 Werte speichern (Sliding Window)
	if len(s.speeds) > 30 {
		s.speeds = s.speeds[1:]
	}
	s.mu.Unlock()
}

// median berechnet den Median der gespeicherten Geschwindigkeiten.
// Gibt 0 zurück wenn weniger als 5 Werte vorhanden sind.
func (s *speedTracker) median() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Mindestens 5 Werte für zuverlässige Berechnung
	if len(s.speeds) < 5 {
		return 0
	}

	sorted := make([]float64, len(s.speeds))
	copy(sorted, s.speeds)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}
