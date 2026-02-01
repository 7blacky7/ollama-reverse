// runner_types.go - Typen und Strukturen fuer den Ollama Runner
//
// Enthaelt:
// - response: Antwort-Container fuer generierte Texte
// - Sequence: Verwaltung einer einzelnen Inferenz-Sequenz
// - NewSequenceParams: Parameter fuer neue Sequenzen
// - batchState: Zustand fuer Batch-Verarbeitung
// - Server: Haupt-Server-Struktur

package ollamarunner

import (
	"hash/maphash"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/sample"
)

// response enthaelt ein generiertes Textstueck mit optionalen Logprobs
type response struct {
	content  string
	logprobs []llm.Logprob
}

// Sequence repraesentiert eine einzelne Inferenz-Sequenz
type Sequence struct {
	// ctxs werden fuer Tensoren verwendet, die die gesamte Lebensdauer der Sequenz ueberdauern
	ctxs []ml.Context

	// mmStore speichert multimodale Embeddings fuer Memory-Management
	mmStore multimodalStore

	// batch index
	iBatch int

	// Prompt-Inputs die noch verarbeitet werden muessen
	inputs []*input.Input

	// Inputs die zum Batch hinzugefuegt aber noch nicht an Forward uebergeben wurden
	pendingInputs []*input.Input

	// Generierte Tokens die noch nicht zurueckgegeben wurden (z.B. fuer Stop-Sequenzen)
	pendingResponses []string

	// Logprobs fuer noch nicht zurueckgegebene Tokens
	pendingLogprobs []llm.Logprob

	// Input-Cache der von dieser Sequenz verwendet wird
	cache *InputCacheSlot

	// Kanal zum Senden von Antworten
	responses chan response

	// Kanal zum Stoppen der Decodierung (z.B. wenn die Verbindung geschlossen wird)
	quit chan bool

	// Anzahl der zu generierenden Tokens
	numPredict int

	// Sampler mit Transformationen fuer generierte Logits
	sampler sample.Sampler

	// Kanal fuer Embeddings (nur wenn embeddingOnly)
	embedding chan []float32

	// Stop-Sequenzen
	stop []string

	// Anzahl der Inputs die beim Context-Shift behalten werden sollen
	numKeep int32

	// True wenn Embeddings statt Textgenerierung zurueckgegeben werden sollen
	embeddingOnly bool

	// Context-Window verschieben wenn ueberschritten
	shift bool

	doneReason llm.DoneReason

	// Logprobs-Konfiguration
	logprobs    bool
	topLogprobs int

	// Metriken
	startedAt, lastUpdatedAt time.Time
	processingDuration       time.Duration
	samplingDuration         time.Duration
	numPredicted             int
	numPromptInputs          int
}

// NewSequenceParams enthaelt Parameter fuer die Erstellung neuer Sequenzen
type NewSequenceParams struct {
	numPredict  int
	stop        []string
	numKeep     int32
	sampler     sample.Sampler
	embedding   bool
	shift       bool
	truncate    bool
	logprobs    bool
	topLogprobs int
}

// batchState verwaltet den Zustand zwischen forwardBatch, computeBatch und predictForwardBatch
type batchState struct {
	// id ist ein Zaehler fuer Trace-Logging
	id int

	// ctx haelt den Backend-Context fuer diesen Batch
	ctx ml.Context

	// modelOutput enthaelt die Ausgaben dieses Batches
	modelOutput ml.Tensor

	// batchInputs haelt die Input-Token-Pointer (anfangs Platzhalter)
	batchInputs []*input.Input

	// batch enthaelt die Inputs fuer einen Model Forward Pass
	batch input.Batch

	// Vollstaendige Menge der Sequenzen zum Zeitpunkt der Batch-Initiierung
	seqs []*Sequence

	// Signal wenn Inputs bereit sind und Compute starten kann
	inputsReadyCh chan struct{}

	// Signal wenn Compute beginnt und seqs fuer naechsten Batch vorbereitet sind
	computeStartedCh chan struct{}

	// Signal wenn Outputs fertig sind und naechster Batch starten kann
	outputsReadyCh chan struct{}
}

// Server ist die Haupt-Struktur fuer den Ollama Runner
type Server struct {
	// modelPath ist der Pfad zur Model-Datei
	modelPath string

	// loadMu verhindert mehrfache gleichzeitige Ladeversuche
	loadMu sync.Mutex

	// lastLoad ist der letzte Ladeversuch (fuer Memory-Wiederverwendung)
	lastLoad llm.LoadRequest

	// ready zeigt an ob der Server bereit ist
	ready sync.WaitGroup

	// Geladenes Model
	model model.Model

	// Status fuer externe Health-Checks
	status llm.ServerStatus

	// Aktueller Ladefortschritt
	progress float32

	// Anzahl gleichzeitiger Requests
	parallel int

	// Maximale Batch-Groesse (pro Sequenz)
	batchSize int

	// Zaehler fuer Trace-Logging
	batchID int

	// Schuetzt Zugriff auf alles darunter
	mu sync.Mutex

	// Signalisiert dass Daten zur Verarbeitung bereit sind
	cond *sync.Cond

	// Liste der gleichzeitig verarbeiteten Sequenzen
	seqs []*Sequence

	// Semaphore fuer maximale Anzahl paralleler Sequenzen
	seqsSem *semaphore.Weighted

	// KV-Cache
	cache *InputCache

	// Naechste Sequenz fuer Prompt-Verarbeitung (verhindert Starvation)
	nextSeq int

	// Hash-Generator fuer multimodale Daten
	multimodalHash maphash.Hash
}
