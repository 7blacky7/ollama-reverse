// backend.go - Backend-Struktur und Basis-Methoden
// Enthält: Backend struct, init(), Close(), einfache Getter

package ggml

// #cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
// #cgo windows LDFLAGS: -lpthread
// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"sync"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

var (
	cpus, accels, gpus []C.ggml_backend_dev_t
	backends           map[C.ggml_backend_dev_t]C.ggml_backend_t
)

// initDevices initialisiert alle verfügbaren Backend-Geräte (einmalig)
var initDevices = sync.OnceFunc(func() {
	ggml.OnceLoad()

	backends = make(map[C.ggml_backend_dev_t]C.ggml_backend_t)
	for i := range C.ggml_backend_dev_count() {
		d := C.ggml_backend_dev_get(i)

		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU:
			if len(cpus) == 0 {
				// Nur das erste CPU-Gerät verwenden
				cpus = append(cpus, d)
			}
		case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			accels = append(accels, d)
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			gpus = append(gpus, d)
		}

		backends[d] = C.ggml_backend_dev_init(d, nil)
	}
})

// layerDevice speichert Gerät und Buffer-Typ für einen Layer
type layerDevice struct {
	d  C.ggml_backend_dev_t
	bt C.ggml_backend_buffer_type_t
}

// Backend ist die GGML-Backend-Implementierung für ML-Operationen
type Backend struct {
	// modelPath ist der Pfad zur Modelldatei
	modelPath string

	meta *fsggml.GGML

	// allocMemory bedeutet, dass Speicher für Tensoren allokiert werden soll
	allocMemory bool

	// tensorLoadTargets mappt Tensor-Namen aus der Datei auf Model-Namen
	tensorLoadTargets map[string][]string

	schedMu       sync.Mutex // Nur ein Compute kann gleichzeitig laufen
	sched         C.ggml_backend_sched_t
	schedBackends []C.ggml_backend_t
	schedBufts    []C.ggml_backend_buffer_type_t

	tensors map[string]*C.struct_ggml_tensor

	// input ist der Buffer-Typ für Eingaben
	input C.ggml_backend_buffer_type_t

	// output ist das Backend-Gerät für Ausgaben
	output C.ggml_backend_dev_t

	// layers ist das Backend für wiederholende Layer
	layers map[int]layerDevice

	// requiredMemory ist der kumulative Speicherbedarf
	requiredMemory *ml.BackendMemory

	// btDeviceMemory mappt Buffer-Typen auf Gerätespeicher
	btDeviceMemory map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory

	flashAttention ml.FlashAttentionType

	// maxGraphNodes ist die maximale Anzahl an Graph-Knoten
	maxGraphNodes int

	// weightBuffers sind GGML-Kontexte und Buffer für Gewichte
	weightBuffers map[*C.struct_ggml_context]C.ggml_backend_buffer_t
}

func init() {
	ml.RegisterBackend("ggml", New)
}

// Close gibt alle Backend-Ressourcen frei
func (b *Backend) Close() {
	if b == nil {
		return
	}

	for ctx, buf := range b.weightBuffers {
		C.ggml_backend_buffer_free(buf)
		C.ggml_free(ctx)
	}

	C.ggml_backend_sched_free(b.sched)
}

// BackendMemory gibt den Speicherbedarf zurück
func (b *Backend) BackendMemory() ml.BackendMemory {
	return *b.requiredMemory
}

// Config gibt die Modell-Konfiguration zurück
func (b *Backend) Config() fs.Config {
	return b.meta.KV()
}

// Get gibt einen Tensor nach Namen zurück
func (b *Backend) Get(name string) ml.Tensor {
	if t, ok := b.tensors[name]; ok {
		return &Tensor{b: b, t: t}
	}

	return nil
}

// NewContext erstellt einen neuen Kontext mit maximaler Graph-Größe
func (b *Backend) NewContext() ml.Context {
	return b.NewContextSize(b.maxGraphNodes)
}

// NewContextSize erstellt einen neuen Kontext mit angegebener Größe
func (b *Backend) NewContextSize(n int) ml.Context {
	if n > b.maxGraphNodes {
		panic("requested number of graph nodes exceeds maximum")
	}

	var allocatedBuffers []C.ggml_backend_buffer_t

	return &Context{
		b:             b,
		maxGraphNodes: n,
		ctx: C.ggml_init(C.struct_ggml_init_params{
			mem_size: C.size_t(n)*C.ggml_tensor_overhead() + C.ggml_graph_overhead_custom(C.size_t(n), false),
			no_alloc: true,
		}),
		allocatedBuffers: &allocatedBuffers,
		layer:            -1,
	}
}

// CacheConfig gibt die Cache-Konfiguration zurück
func (b *Backend) CacheConfig() ml.CacheConfig {
	if b.flashAttention == ml.FlashAttentionEnabled {
		return ml.CacheConfig{CachePadding: 256, MaskDType: ml.DTypeF16}
	}
	return ml.CacheConfig{CachePadding: 256, PermutedV: true}
}

// pad rundet auf das nächste Vielfache von pad auf
func pad(length, padSize C.size_t) C.size_t {
	return ((length + padSize - 1) / padSize) * padSize
}
