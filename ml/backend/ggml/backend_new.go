// backend_new.go - Backend-Konstruktor
// Enthält: New() Funktion zum Erstellen eines neuen GGML-Backends

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"

import (
	"log/slog"
	"maps"
	"os"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/format"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

var once sync.Once

// deviceBufferType speichert Gerät und zugehörige Buffer-Typen
type deviceBufferType struct {
	d   C.ggml_backend_dev_t
	bts []C.ggml_backend_buffer_type_t
}

// New erstellt ein neues GGML-Backend für das angegebene Modell
func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	once.Do(func() {
		slog.Info(
			"",
			"architecture", meta.KV().Architecture(),
			"file_type", meta.KV().FileType(),
			"name", meta.KV().String("general.name"),
			"description", meta.KV().String("general.description"),
			"num_tensors", len(meta.Tensors().Items()),
			"num_key_values", len(meta.KV()),
		)
	})

	initDevices()

	var requiredMemory ml.BackendMemory
	btDeviceMemory := make(map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory)

	blocks := int(meta.KV().BlockCount())

	// Buffer-Typen für CPU und GPU erstellen
	cpuDeviceBufferType, gpuDeviceBufferTypes := setupDeviceBufferTypes(blocks, &requiredMemory, btDeviceMemory)

	// Layer-Zuweisung
	input := cpuDeviceBufferType
	assignLayer := createLayerAssigner(params, &requiredMemory, gpuDeviceBufferTypes, cpuDeviceBufferType)

	layers := make([]deviceBufferType, blocks)
	for i := range layers {
		layers[i] = assignLayer(i)
	}
	output := assignLayer(blocks)

	// Tensoren erstellen
	maxTensors := len(meta.Tensors().Items()) + 1 + blocks*2
	targets, ctxs := createTensors(meta, input, output, layers, blocks, maxTensors, &requiredMemory, btDeviceMemory)

	// Tensor-Map erstellen
	tensors := buildTensorMap(ctxs)

	// Scheduler erstellen
	deviceBufferTypes, schedBackends, schedBufts := setupScheduler(ctxs, cpuDeviceBufferType, params)

	maxGraphNodes := max(1024, len(meta.Tensors().Items())*8)
	sched := C.ggml_backend_sched_new_ext(
		(*C.ggml_backend_t)(unsafe.Pointer(&schedBackends[0])),
		(*C.ggml_backend_buffer_type_t)(unsafe.Pointer(&schedBufts[0])),
		C.int(len(schedBackends)),
		C.size_t(maxGraphNodes),
		C._Bool(false),
		C._Bool(true),
		C._Bool(params.AllocMemory),
	)

	// Buffer allokieren
	bbs := allocateBuffers(ctxs, requiredMemory)

	for bs := range maps.Values(bbs) {
		logutil.Trace("model weights", "buffer", C.GoString(C.ggml_backend_buffer_name(bs)),
			"size", format.HumanBytes2(uint64(C.ggml_backend_buffer_get_size(bs))))
	}

	return &Backend{
		modelPath:         modelPath,
		allocMemory:       params.AllocMemory,
		flashAttention:    params.FlashAttention,
		meta:              meta,
		tensorLoadTargets: targets,
		tensors:           tensors,
		sched:             sched,
		schedBackends:     schedBackends,
		schedBufts:        schedBufts,
		input:             deviceBufferTypes[input.d],
		output:            output.d,
		layers:            buildLayerMap(layers, deviceBufferTypes),
		requiredMemory:    &requiredMemory,
		btDeviceMemory:    btDeviceMemory,
		maxGraphNodes:     maxGraphNodes,
		weightBuffers:     bbs,
	}, nil
}
