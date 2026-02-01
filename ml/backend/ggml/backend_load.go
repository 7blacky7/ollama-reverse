// backend_load.go - Model Loading und Device-Infos
// Enthält: Load(), BackendDevices()

package ggml

// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"unsafe"

	"github.com/ollama/ollama/format"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	"golang.org/x/sync/errgroup"
)

// Load lädt die Modellgewichte in den Speicher
func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	if !b.allocMemory {
		return errors.New("cannot load model without memory allocation")
	}

	// Log-Ausgabe über GPU-Layer-Offloading
	gpuLayers := 0
	for layer := range maps.Values(b.layers) {
		switch C.ggml_backend_dev_type(layer.d) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			gpuLayers++
		}
	}
	slog.Info(fmt.Sprintf("offloading %d repeating layers to GPU", gpuLayers))

	switch C.ggml_backend_dev_type(b.output) {
	case C.GGML_BACKEND_DEVICE_TYPE_CPU:
		slog.Info("offloading output layer to CPU")
	case C.GGML_BACKEND_DEVICE_TYPE_GPU,
		C.GGML_BACKEND_DEVICE_TYPE_IGPU:
		slog.Info("offloading output layer to GPU")
		gpuLayers++
	case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
		slog.Info("offloading output layer to ACCEL")
	}
	slog.Info(fmt.Sprintf("offloaded %d/%d layers to GPU", gpuLayers, len(b.layers)+1))

	var doneBytes atomic.Uint64
	totalBytes := uint64(b.meta.Length) - b.meta.Tensors().Offset

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, t := range b.meta.Tensors().Items() {
		g.Go(func() error {
			return b.loadTensor(ctx, t, &doneBytes, totalBytes, progress)
		})
	}

	// Ungenutzte Backend-Devices aufräumen
	b.cleanupUnusedDevices()

	if err := g.Wait(); err != nil {
		return err
	}

	return nil
}

// loadTensor lädt einen einzelnen Tensor
func (b *Backend) loadTensor(ctx context.Context, t *fsggml.Tensor, doneBytes *atomic.Uint64, totalBytes uint64, progress func(float32)) error {
	tts := make([]*C.struct_ggml_tensor, max(1, len(b.tensorLoadTargets[t.Name])))
	for i := range tts {
		target := b.tensorLoadTargets[t.Name][i]
		if target == "" {
			target = t.Name
		}

		tt, ok := b.tensors[target]
		if !ok {
			return fmt.Errorf("unassigned tensor: %s", t.Name)
		}

		tts[i] = tt
	}

	// Eigener FD für jede Goroutine für sequentielles Lesen
	file, err := os.Open(b.modelPath)
	if err != nil {
		slog.Warn("file open error", "file", b.modelPath, "error", err)
		return err
	}
	defer file.Close()
	sr := io.NewSectionReader(file, int64(b.meta.Tensors().Offset+t.Offset), int64(t.Size()))

	// MXFP4 Konvertierung
	if t.Kind == 4 && tts[0]._type == 39 {
		return b.loadMXFP4Tensor(ctx, sr, t, tts, doneBytes, totalBytes, progress)
	}

	// BF16 zu FP32 Konvertierung
	if strings.HasSuffix(t.Name, "_exps.bias") && t.Kind == 30 && tts[0]._type == 0 {
		return b.loadBF16ToFP32Tensor(ctx, sr, t, tts, doneBytes, totalBytes, progress)
	}

	// Standard-Tensor laden
	return b.loadStandardTensor(ctx, sr, t, tts, doneBytes, totalBytes, progress)
}

// loadMXFP4Tensor lädt MXFP4-formatierte Tensoren
func (b *Backend) loadMXFP4Tensor(ctx context.Context, sr *io.SectionReader, t *fsggml.Tensor, tts []*C.struct_ggml_tensor, doneBytes *atomic.Uint64, totalBytes uint64, progress func(float32)) error {
	const BS = 17                             // MXFP4 block size
	bts := make([]byte, 8*BS*format.KibiByte) // ~128k block aligned
	var s uint64
	var tmp [16]byte
	for s < t.Size() {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
		if err != nil {
			slog.Warn("file read error", "file", b.modelPath, "error", err)
			return err
		}
		for j := range n / BS {
			for i := 1; i < 9; i++ {
				// Transformiere a1b2c3 ... x7y8z9 -> 71xa82yb93zc
				a, b := bts[j*BS+i], bts[j*BS+i+8]
				tmp[2*(i-1)] = (a & 0x0F) | (b << 4)
				tmp[2*(i-1)+1] = (a >> 4) | (b & 0xF0)
			}
			copy(bts[j*BS+1:j*BS+17], tmp[:])
		}

		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
		}

		s += uint64(n)

		if progress != nil {
			done := doneBytes.Add(uint64(n))
			progress(float32(done) / float32(totalBytes))
		}
	}
	return nil
}

// loadBF16ToFP32Tensor lädt BF16-Tensoren und konvertiert zu FP32
func (b *Backend) loadBF16ToFP32Tensor(ctx context.Context, sr *io.SectionReader, t *fsggml.Tensor, tts []*C.struct_ggml_tensor, doneBytes *atomic.Uint64, totalBytes uint64, progress func(float32)) error {
	bts := make([]byte, 128*format.KibiByte)
	var e uint64
	for e < t.Elements() {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Elements()-e)*2)])
		if err != nil {
			slog.Warn("file read error", "file", b.modelPath, "error", err)
			return err
		}
		fp32 := ConvertToF32(bts, uint32(fsggml.TensorTypeBF16), uint64(n/2))

		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&fp32[0]), C.size_t(e*4), C.size_t(n*2))
		}
		e += uint64(n / 2)
		if progress != nil {
			done := doneBytes.Add(uint64(n))
			progress(float32(done) / float32(totalBytes))
		}
	}
	return nil
}

// loadStandardTensor lädt Standard-Tensoren
func (b *Backend) loadStandardTensor(ctx context.Context, sr *io.SectionReader, t *fsggml.Tensor, tts []*C.struct_ggml_tensor, doneBytes *atomic.Uint64, totalBytes uint64, progress func(float32)) error {
	bts := make([]byte, 128*format.KibiByte)

	var s uint64
	for s < t.Size() {
		if err := ctx.Err(); err != nil {
			return err
		}

		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
		if err != nil {
			slog.Warn("file read error", "file", b.modelPath, "error", err)
			return err
		}

		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
		}

		s += uint64(n)

		if progress != nil {
			done := doneBytes.Add(uint64(n))
			progress(float32(done) / float32(totalBytes))
		}
	}

	return nil
}

// cleanupUnusedDevices räumt ungenutzte Backend-Devices auf
func (b *Backend) cleanupUnusedDevices() {
nextDevice:
	for _, d := range append(gpus, append(accels, cpus...)...) {
		for _, backend := range b.schedBackends {
			if d == C.ggml_backend_get_device(backend) {
				continue nextDevice
			}
		}

		C.ggml_backend_dev_reset(d)
	}
}

// BackendDevices gibt Informationen über verfügbare Geräte zurück
func (b *Backend) BackendDevices() []ml.DeviceInfo {
	deviceInfos := []ml.DeviceInfo{}
	for _, dev := range gpus {
		// Überspringe ungenutzte Geräte wenn ein Modell geladen ist
		if b.allocMemory {
			idleDev := true
			for _, backend := range b.schedBackends {
				if dev == C.ggml_backend_get_device(backend) {
					idleDev = false
					break
				}
			}
			if idleDev {
				slog.Debug("skipping unused backend device", "description", C.GoString(C.ggml_backend_dev_description(dev)))
				continue
			}
		}

		info := ml.DeviceInfo{}
		props := C.struct_ggml_backend_dev_props{}
		C.ggml_backend_dev_get_props(dev, &props)
		info.Name = C.GoString(props.name)
		info.Description = C.GoString(props.description)
		info.ID = C.GoString(props.id)
		info.Library = C.GoString(props.library)
		info.ComputeMajor = (int)(props.compute_major)
		info.ComputeMinor = (int)(props.compute_minor)
		info.DriverMajor = (int)(props.driver_major)
		info.DriverMinor = (int)(props.driver_minor)
		info.Integrated = props.integrated != 0
		if props.library != nil {
			info.Library = C.GoString(props.library)
		}
		if props.device_id != nil {
			info.PCIID = C.GoString(props.device_id)
		}
		info.LibraryPath = ggml.LibPaths()
		C.ggml_backend_dev_memory(dev, &props.memory_free, &props.memory_total)
		info.TotalMemory = (uint64)(props.memory_total)
		info.FreeMemory = (uint64)(props.memory_free)

		deviceInfos = append(deviceInfos, info)
	}
	return deviceInfos
}
