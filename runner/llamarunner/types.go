// Package llamarunner - LLM Runner Server
//
// Dieses Modul definiert die Kerntypen für den LLM-Runner:
// - response: Generierter Text mit optionalen Logprobs
// - input: Prompt-Element (Token oder Bild-Embedding)
// - Sequence: Repräsentiert eine laufende Inferenz-Sequenz
// - Server: HTTP-Server für LLM-Inferenz
package llamarunner

import (
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
)

// response enthält generiertes Text-Fragment mit optionalen Logprobs
type response struct {
	content  string
	logprobs []llm.Logprob
}

// input ist ein Element des Prompts - entweder ein Token oder ein Bild-Embedding
type input struct {
	token int

	// embed ist ein Bild-Embedding (von Vision Projector generiert)
	embed []float32
}

// Sequence repräsentiert eine laufende Inferenz-Sequenz
type Sequence struct {
	// iBatch ist der Batch-Index
	iBatch int

	// numPredicted zählt die bisher generierten Tokens
	numPredicted int

	// inputs sind die noch zu verarbeitenden Prompt-Inputs
	inputs []input

	// pendingInputs wurden zum Batch hinzugefügt, aber noch nicht dekodiert
	pendingInputs []input

	// pendingResponses sind generierte Tokens, die noch nicht zurückgegeben wurden
	pendingResponses []string

	// pendingLogprobs für noch nicht zurückgegebene Tokens
	pendingLogprobs []llm.Logprob

	// cache ist der Input-Cache-Slot für diese Sequenz
	cache *InputCacheSlot

	// responses ist der Kanal für Antworten
	responses chan response

	// quit signalisiert Abbruch (z.B. bei geschlossener Verbindung)
	quit chan bool

	// numPredict ist die maximale Anzahl zu generierender Tokens
	numPredict int

	// samplingCtx enthält den Sampling-Kontext
	samplingCtx *llama.SamplingContext

	// embedding Kanal für Embedding-Ergebnisse
	embedding chan []float32

	// stop enthält die Stop-Sequenzen
	stop []string

	// numKeep Anzahl der Inputs am Anfang die beim Context-Shift behalten werden
	numKeep int

	// embeddingOnly gibt an ob nur Embeddings generiert werden sollen
	embeddingOnly bool

	// shift erlaubt Context-Window-Verschiebung bei Überschreitung
	shift bool

	// doneReason enthält den Grund für das Ende der Generierung
	doneReason llm.DoneReason

	// logprobs Konfiguration
	logprobs    bool
	topLogprobs int

	// Metriken
	processingDuration time.Duration
	generationDuration time.Duration
	numDecoded         int
	numPromptInputs    int
}

// NewSequenceParams enthält Parameter für neue Sequenzen
type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
	shift          bool
	truncate       bool
	logprobs       bool
	topLogprobs    int
}

// Server ist der HTTP-Server für LLM-Inferenz
type Server struct {
	// modelPath ist der Pfad zur Modell-Datei
	modelPath string

	// loadMu verhindert gleichzeitige Lade-Versuche
	loadMu sync.Mutex

	// ready signalisiert dass der Server bereit ist
	// schützt Zugriff auf model und image
	ready sync.WaitGroup

	// model ist das geladene LLM-Modell
	model *llama.Model

	// image ist der Bildkontext für multimodale Modelle
	image *ImageContext

	// status für externe Health-Checks
	status llm.ServerStatus

	// progress zeigt den aktuellen Lade-Fortschritt
	progress float32

	// parallel ist die Anzahl gleichzeitiger Requests
	parallel int

	// batchSize ist die maximale Batch-Größe pro Sequenz
	batchSize int

	// mu schützt den Zugriff auf den Dekodier-Zustand
	mu sync.Mutex

	// cond signalisiert neue Daten zur Verarbeitung
	cond *sync.Cond

	// lc ist der LLaMA-Dekodier-Kontext
	lc *llama.Context

	// seqs enthält die gleichzeitig laufenden Sequenzen
	seqs []*Sequence

	// seqsSem begrenzt parallele Sequenzen
	seqsSem *semaphore.Weighted

	// cache ist der KV-Cache
	cache *InputCache

	// nextSeq ist die nächste Sequenz für Prompt-Verarbeitung (gegen Starvation)
	nextSeq int
}

// allNil prüft ob alle Sequenzen nil sind
func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}
