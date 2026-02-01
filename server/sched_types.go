// Package server - Scheduler Typen und Strukturen
//
// Diese Datei enthält:
// - LlmRequest: Anfrage-Struktur für Model-Laden
// - Scheduler: Haupt-Scheduler-Struktur
// - runnerRef: Referenz auf geladenen Runner
// - Sortier-Typen für Runner-Auswahl
package server

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Standardwert für automatische Model-Anzahl pro GPU
// Models müssen trotzdem ins VRAM passen
var defaultModelsPerGPU = 3

// ErrMaxQueue wird zurückgegeben wenn die Warteschlange voll ist
var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

// LlmRequest repräsentiert eine Anfrage zum Laden eines Models
type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint
}

// Scheduler verwaltet das Laden und Entladen von Models
type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan any

	// loadedMu schützt loaded und activeLoading
	loadedMu sync.Mutex

	// activeLoading ist das Model das gerade geladen wird,
	// inklusive dem Entladen anderer Models. Nur ein Model kann
	// gleichzeitig geladen werden, aber Requests für bereits geladene
	// Models können parallel bearbeitet werden
	activeLoading llm.LlamaServer
	loaded        map[string]*runnerRef

	loadFn          func(req *LlmRequest, f *ggml.GGML, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error)
	getGpuFn        func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo
	getSystemInfoFn func() ml.SystemInfo
	waitForRecovery time.Duration
}

// InitScheduler erstellt einen neuen Scheduler
func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pendingReqCh:    make(chan *LlmRequest, maxQueue),
		finishedReqCh:   make(chan *LlmRequest, maxQueue),
		expiredCh:       make(chan *runnerRef, maxQueue),
		unloadedCh:      make(chan any, maxQueue),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     llm.NewLlamaServer,
		getGpuFn:        discover.GPUDevices,
		getSystemInfoFn: discover.GetSystemInfo,
		waitForRecovery: 5 * time.Second,
	}
	sched.loadFn = sched.load
	return sched
}

// runnerRef hält eine Referenz auf einen geladenen Model-Runner
type runnerRef struct {
	refMu    sync.Mutex
	refCount uint // Verhindert Entladen wenn > 0

	llama        llm.LlamaServer
	pid          int
	loading      bool          // True nur während initialem Laden
	gpus         []ml.DeviceID // Aufgezeichnet bei Provisionierung
	discreteGPUs bool          // True wenn alle Devices diskrete GPUs sind - für VRAM-Recovery-Check bei iGPUs
	vramSize     uint64
	totalSize    uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time

	model       *Model
	modelPath   string
	numParallel int
	*api.Options
}

// LogValue formatiert den Runner für Logging
func (runner *runnerRef) LogValue() slog.Value {
	if runner == nil {
		return slog.StringValue("nil")
	}
	attrs := []slog.Attr{}
	if runner.model != nil {
		attrs = append(attrs, slog.String("name", runner.model.Name))
	}
	if len(runner.gpus) > 0 {
		attrs = append(attrs,
			slog.Any("inference", runner.gpus),
		)
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.Int("pid", runner.pid),
		slog.String("model", runner.modelPath),
	)
	if runner.Options != nil {
		attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
	}
	return slog.GroupValue(attrs...)
}

// unload entlädt den Runner. refMu muss bereits gehalten werden!
func (runner *runnerRef) unload() {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.model = nil
	runner.Options = nil
	runner.gpus = nil
}

// Implementiert discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	if runner.llama != nil {
		return runner.llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if runner.llama != nil {
		return runner.llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	return runner.gpus
}

func (runner *runnerRef) HasExited() bool {
	if runner.llama != nil {
		return runner.llama.HasExited()
	}
	return true
}

// ByDurationAndName sortiert Runner nach Session-Dauer und Name
type ByDurationAndName []*runnerRef

func (a ByDurationAndName) Len() int      { return len(a) }
func (a ByDurationAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDurationAndName) Less(i, j int) bool {
	// Primär nach Session-Dauer (uint64 für negative Werte)
	d1 := uint64(a[i].sessionDuration)
	d2 := uint64(a[j].sessionDuration)
	if d1 != d2 {
		return d1 < d2
	}
	// Sekundär nach Model-Pfad lexikografisch
	return a[i].modelPath < a[j].modelPath
}
