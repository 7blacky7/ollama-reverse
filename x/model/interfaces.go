// Package model definiert die Kern-Interfaces und Typen fuer Modelle.
//
// Diese Datei enthaelt:
// - Model Interface (Forward, Backend, Config)
// - MultimodalProcessor Interface (EncodeMultimodal, PostTokenize)
// - Base Struct (Basisimplementierung fuer alle Modelle)
// - config Struct (KV-Cache Konfiguration)
package model

import (
	"github.com/ollama/ollama/x/kvcache"
	"github.com/ollama/ollama/x/ml"
	"github.com/ollama/ollama/x/model/input"
)

// ============================================================================
// Interfaces
// ============================================================================

// Model implementiert eine spezifische Modell-Architektur.
// Definiert den Forward-Pass und modellspezifische Konfiguration.
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
}

// MultimodalProcessor muss von multimodalen Modellen implementiert werden.
type MultimodalProcessor interface {
	// EncodeMultimodal verarbeitet einen einzelnen Input (z.B. ein Bild) und
	// generiert einen Output (typischerweise ein Embedding), das vom Modell
	// verwendet werden kann.
	//
	// Der Rueckgabewert ist ein oder mehrere Tensoren, jeweils mit optionalen
	// modellspezifischen Metadaten. Typischerweise koennten die Tensoren Views
	// in ein Embedding sein, wobei jeder View einen Chunk von Daten repraesentiert,
	// der unabhaengig in verschiedenen Batches verarbeitet werden kann.
	//
	// Das Ergebnis kann vom Runner gecacht werden.
	EncodeMultimodal(ml.Context, []byte) ([]input.Multimodal, error)

	// PostTokenize wird nach der Tokenisierung aufgerufen, um dem Modell zu
	// ermoeglichen, den Input-Stream zu bearbeiten und multimodale Elemente
	// korrekt anzuordnen.
	//
	// Der Input ist ein Slice von Tokens mit den Ergebnissen von EncodeMultimodal
	// in der Reihenfolge, wie sie der Benutzer bereitgestellt hat. Jedes Element
	// des Slice ist entweder ein einzelner Token oder ein einzelnes multimodales Objekt.
	//
	// Das Modell muss sicherstellen, dass Inputs entsprechend ihrer Verarbeitung
	// und Speicherung im Cache angeordnet sind. Zum Beispiel sollten Llava-Style-
	// Modelle Platzhalter-Tokens einfuegen, die der Feature-Groesse des entsprechenden
	// Bildes entsprechen, wobei das Bild selbst an diese Tokens angehaengt und
	// aufgeteilt wird.
	//
	// Diese Funktion ist auch dafuer verantwortlich, MultimodalHash fuer jedes
	// geaenderte Multimodal zu aktualisieren, um sicherzustellen, dass ein
	// eindeutiger Hash-Wert existiert, der den Inhalt genau repraesentiert.
	PostTokenize([]*input.Input) ([]*input.Input, error)
}

// ============================================================================
// Base Struct
// ============================================================================

// Base implementiert die gemeinsamen Felder und Methoden fuer alle Modelle.
type Base struct {
	b ml.Backend
	config
}

// config enthaelt die Modell-Konfiguration.
type config struct {
	Cache kvcache.Cache
}

// Backend gibt das zugrunde liegende Backend zurueck, das das Modell ausfuehrt.
func (m *Base) Backend() ml.Backend {
	return m.b
}

// Config gibt die Modell-Konfiguration zurueck.
func (m *Base) Config() config {
	return m.config
}
