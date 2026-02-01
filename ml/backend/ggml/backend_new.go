// backend_new.go - Backend-Konstruktor
// Enthält: New() Funktion zum Erstellen eines neuen Backends

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
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"
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

// setupDeviceBufferTypes initialisiert CPU und GPU Buffer-Typen
func setupDeviceBufferTypes(blocks int, requiredMemory *ml.BackendMemory, btDeviceMemory map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory) (deviceBufferType, []deviceBufferType) {
	cpuDeviceBufferType := deviceBufferType{d: C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU)}
	for _, d := range append(accels, append(gpus, cpus...)...) {
		switch C.ggml_backend_dev_type(d) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
			bt := C.ggml_backend_dev_buffer_type(d)
			cpuDeviceBufferType.bts = append(cpuDeviceBufferType.bts, bt)
			btDeviceMemory[C.ggml_backend_dev_buffer_type(d)] = &requiredMemory.CPU
		}
	}

	requiredMemory.CPU.Name = C.GoString(C.ggml_backend_dev_name(cpuDeviceBufferType.d))
	var props C.struct_ggml_backend_dev_props
	C.ggml_backend_dev_get_props(cpuDeviceBufferType.d, &props)
	requiredMemory.CPU.ID = C.GoString(props.id)
	requiredMemory.CPU.Library = C.GoString(props.library)
	requiredMemory.CPU.Weights = make([]uint64, blocks+1)
	requiredMemory.CPU.Cache = make([]uint64, blocks+1)

	var gpuDeviceBufferTypes []deviceBufferType
	requiredMemory.GPUs = make([]ml.DeviceMemory, len(gpus))
	for i, d := range gpus {
		bt := C.ggml_backend_dev_buffer_type(d)
		gpuDeviceBufferTypes = append(gpuDeviceBufferTypes, deviceBufferType{
			d:   d,
			bts: append([]C.ggml_backend_buffer_type_t{bt}, cpuDeviceBufferType.bts...),
		})

		btDeviceMemory[bt] = &requiredMemory.GPUs[i]
		requiredMemory.GPUs[i].Name = C.GoString(C.ggml_backend_dev_name(d))
		C.ggml_backend_dev_get_props(d, &props)
		requiredMemory.GPUs[i].ID = C.GoString(props.id)
		requiredMemory.GPUs[i].Library = C.GoString(props.library)
		requiredMemory.GPUs[i].Weights = make([]uint64, blocks+1)
		requiredMemory.GPUs[i].Cache = make([]uint64, blocks+1)
	}

	return cpuDeviceBufferType, gpuDeviceBufferTypes
}

// createLayerAssigner erstellt eine Funktion zur Layer-Zuweisung
func createLayerAssigner(params ml.BackendParams, requiredMemory *ml.BackendMemory, gpuDeviceBufferTypes []deviceBufferType, cpuDeviceBufferType deviceBufferType) func(int) deviceBufferType {
	return func(layer int) deviceBufferType {
		for _, p := range params.GPULayers {
			for _, l := range p.Layers {
				if l == layer {
					for i := range requiredMemory.GPUs {
						if requiredMemory.GPUs[i].DeviceID == p.DeviceID {
							return gpuDeviceBufferTypes[i]
						}
					}
					return cpuDeviceBufferType
				}
			}
		}
		return cpuDeviceBufferType
	}
}

// createTensors erstellt alle Tensoren aus den Metadaten
func createTensors(meta *fsggml.GGML, input, output deviceBufferType, layers []deviceBufferType, blocks, maxTensors int, requiredMemory *ml.BackendMemory, btDeviceMemory map[C.ggml_backend_buffer_type_t]*ml.DeviceMemory) (map[string][]string, map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context) {
	type tensor struct {
		source *fsggml.Tensor
		target string
	}

	targets := make(map[string][]string)
	ctxs := make(map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context)

	createTensor := func(t tensor, bts []C.ggml_backend_buffer_type_t, layer int) *C.struct_ggml_tensor {
		for _, bt := range bts {
			if _, ok := ctxs[bt]; !ok {
				ctxs[bt] = C.ggml_init(C.struct_ggml_init_params{
					mem_size: C.ggml_tensor_overhead() * C.size_t(maxTensors),
					no_alloc: true,
				})
			}

			targets[t.source.Name] = append(targets[t.source.Name], t.target)

			name := t.source.Name
			if t.target != "" {
				name = t.target
			}

			cname := C.CString(name)
			defer C.free(unsafe.Pointer(cname))
			if tt := C.ggml_get_tensor(ctxs[bt], cname); tt != nil {
				return tt
			}

			kind := t.source.Kind
			if t.source.Kind == 4 {
				kind = 39
			} else if t.source.Kind == uint32(fsggml.TensorTypeBF16) && strings.HasSuffix(t.source.Name, "_exps.bias") {
				kind = uint32(fsggml.TensorTypeF32)
			}

			tt := C.ggml_new_tensor(ctxs[bt], kind, C.int(len(t.source.Shape)), (*C.int64_t)(unsafe.Pointer(&t.source.Shape[0])))
			C.ggml_set_name(tt, cname)

			logutil.Trace("created tensor", "name", name, "shape", t.source.Shape, "dtype", t.source.Kind, "buffer_type", C.GoString(C.ggml_backend_buft_name(bt)))

			size := pad(C.ggml_backend_buft_get_alloc_size(bt, tt), C.ggml_backend_buft_get_alignment(bt))
			if layer == -1 {
				requiredMemory.InputWeights += uint64(size)
			} else {
				btDeviceMemory[bt].Weights[layer] += uint64(size)
			}

			return tt
		}
		return nil
	}

	contains := func(s string, parts ...string) bool {
		split := strings.Split(s, ".")
		for _, part := range parts {
			if slices.Contains(split, part) {
				return true
			}
		}
		return false
	}

	for _, t := range meta.Tensors().Items() {
		switch {
		case contains(t.Name, "position_embd", "token_embd", "token_norm_embd", "token_types"):
			createTensor(tensor{source: t}, input.bts, -1)
			if _, ok := meta.Tensors().GroupLayers()["output"]; !ok && t.Name == "token_embd.weight" {
				createTensor(tensor{source: t, target: "output.weight"}, output.bts, blocks)
			}
		case contains(t.Name, "cls", "output", "output_norm", "altup_proj", "altup_unembd_proj", "per_layer_token_embd", "per_layer_model_proj", "per_layer_proj_norm"):
			createTensor(tensor{source: t}, output.bts, blocks)
		case strings.HasPrefix(t.Name, "v.") || strings.HasPrefix(t.Name, "mm.") || strings.HasPrefix(t.Name, "s."):
			createTensor(tensor{source: t}, output.bts, blocks)
		case contains(t.Name, "rope_freqs", "rope_factors_long", "rope_factors_short"):
			for i, layer := range layers {
				createTensor(tensor{source: t, target: "blk." + strconv.Itoa(i) + "." + t.Name}, layer.bts, i)
			}
		default:
			layerIndex := -1
			if fields := strings.FieldsFunc(t.Name, func(r rune) bool { return !unicode.IsNumber(r) }); len(fields) > 0 {
				if i, err := strconv.Atoi(fields[0]); err == nil {
					layerIndex = i
				}
			}

			if layerIndex >= 0 {
				createTensor(tensor{source: t}, layers[layerIndex].bts, layerIndex)
			} else {
				createTensor(tensor{source: t}, input.bts, -1)
			}
		}
	}

	return targets, ctxs
}

// buildTensorMap erstellt eine Map von Namen zu Tensoren
func buildTensorMap(ctxs map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context) map[string]*C.struct_ggml_tensor {
	tensors := make(map[string]*C.struct_ggml_tensor)
	for _, c := range ctxs {
		for t := C.ggml_get_first_tensor(c); t != nil; t = C.ggml_get_next_tensor(c, t) {
			tensors[C.GoString(C.ggml_get_name(t))] = t
		}
	}
	return tensors
}

// setupScheduler erstellt den Scheduler
func setupScheduler(ctxs map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context, cpuDeviceBufferType deviceBufferType, params ml.BackendParams) (map[C.ggml_backend_dev_t]C.ggml_backend_buffer_type_t, []C.ggml_backend_t, []C.ggml_backend_buffer_type_t) {
	deviceBufferTypes := make(map[C.ggml_backend_dev_t]C.ggml_backend_buffer_type_t)
	var schedBackends []C.ggml_backend_t
	var schedBufts []C.ggml_backend_buffer_type_t

	for _, d := range append(gpus, append(accels, cpus...)...) {
		b := backends[d]
		bt := C.ggml_backend_get_default_buffer_type(b)

		if !slices.Contains(cpuDeviceBufferType.bts, bt) {
			if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
				continue
			}
		}

		deviceBufferTypes[d] = bt
		schedBackends = append(schedBackends, b)
		schedBufts = append(schedBufts, bt)

		if C.ggml_backend_is_cpu(b) {
			C.ggml_backend_cpu_set_n_threads(b, C.int(Threads(params.NumThreads)))
		}
	}

	return deviceBufferTypes, schedBackends, schedBufts
}

// allocateBuffers allokiert Buffer für alle Kontexte
func allocateBuffers(ctxs map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context, requiredMemory ml.BackendMemory) map[*C.struct_ggml_context]C.ggml_backend_buffer_t {
	bbs := make(map[*C.struct_ggml_context]C.ggml_backend_buffer_t, len(ctxs))
	for bt, c := range ctxs {
		if C.ggml_get_first_tensor(c) == nil {
			continue
		}

		b := C.ggml_backend_alloc_ctx_tensors_from_buft(c, bt)
		if b == nil {
			for _, buf := range bbs {
				C.ggml_backend_buffer_free(buf)
			}
			for _, ctx := range ctxs {
				C.ggml_free(ctx)
			}
			panic(ml.ErrNoMem{BackendMemory: requiredMemory})
		}

		C.ggml_backend_buffer_set_usage(b, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
		bbs[c] = b
	}
	return bbs
}

// buildLayerMap erstellt eine Map von Layer-Index zu LayerDevice
func buildLayerMap(layers []deviceBufferType, deviceBufferTypes map[C.ggml_backend_dev_t]C.ggml_backend_buffer_type_t) map[int]layerDevice {
	m := make(map[int]layerDevice)
	for i, layer := range layers {
		m[i] = layerDevice{
			d:  layer.d,
			bt: deviceBufferTypes[layer.d],
		}
	}
	return m
}
