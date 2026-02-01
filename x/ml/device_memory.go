// device_memory.go
// Dieses Modul enthaelt die Speicher-bezogenen Typen und Funktionen fuer
// die Verwaltung von Geraete-Speicher (VRAM, System-RAM).
// Urspruenglich aus device.go extrahiert.

package ml

import (
	"context"
	"fmt"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/format"
)

// ErrNoMem is returned when panicing due to insufficient memory. It includes
// the attempted memory allocation.
type ErrNoMem struct {
	BackendMemory
}

func (e ErrNoMem) Error() string {
	return fmt.Sprintf("insufficient memory - required allocations: %+v", e.BackendMemory)
}

// Minimal unique device identification
type DeviceID struct {
	// ID is an identifier for the device for matching with system
	// management libraries.  The ID is only unique for other devices
	// using the same Library.
	// This ID represents a "post filtered" view of the enumerated devices
	// if the ID is numeric
	ID string `json:"id"`

	// Library identifies which library is used for the device (e.g. CUDA, ROCm, etc.)
	Library string `json:"backend,omitempty"`
}

// DeviceMemory provides a breakdown of the memory needed
// per device, such as a CPU or GPU.
type DeviceMemory struct {
	DeviceID

	// Name is the name of the device as labeled by the backend. It
	// may not be persistent across instances of the runner.
	Name string

	// Weights is the per-layer memory needed for the model weights.
	Weights []uint64

	// Cache is the per-layer memory needed for the KV cache.
	Cache []uint64

	// Graph is the size of the compute graph. It is not per-layer.
	Graph uint64
}

func sumMemory(mem []uint64) uint64 {
	var sum uint64

	for _, m := range mem {
		sum += m
	}

	return sum
}

// Size returns the total size of the memory required by this device
func (m DeviceMemory) Size() uint64 {
	return sumMemory(m.Weights) + sumMemory(m.Cache) + m.Graph
}

func memoryPresent(mem []uint64) bool {
	return slices.ContainsFunc(mem, func(m uint64) bool { return m != 0 })
}

func (m DeviceMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if memoryPresent(m.Weights) {
		attrs = append(attrs, slog.Any("Weights", m.Weights))
	}

	if memoryPresent(m.Cache) {
		attrs = append(attrs, slog.Any("Cache", m.Cache))
	}

	if m.Graph != 0 {
		attrs = append(attrs, slog.Any("Graph", m.Graph))
	}

	if len(attrs) > 0 && m.ID != "" {
		attrs = append([]slog.Attr{slog.String("ID", m.ID)}, attrs...)
	}

	return slog.GroupValue(attrs...)
}

// BackendMemory provides the amount of memory required to load the model
// per device based on the BackendParams. In some cases, not all required
// allocations will be known at this point. However, the size of the most recent
// allocation is guaranteed to be provided so that if it failed, the caller can
// accommodate that to make forward progress.
type BackendMemory struct {
	// InputWeights are always located on the CPU and cannot be moved
	InputWeights uint64

	// CPU model components are located in system memory. This does not
	// include unified memory allocated through the GPU.
	CPU DeviceMemory

	// GPU model components are located on one or more GPUs.
	GPUs []DeviceMemory
}

func (m BackendMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if m.InputWeights != 0 {
		attrs = append(attrs, slog.Any("InputWeights", m.InputWeights))
	}

	attrs = append(attrs, slog.Any(m.CPU.Name, m.CPU))
	for _, g := range m.GPUs {
		attrs = append(attrs, slog.Any(g.Name, g))
	}

	return slog.GroupValue(attrs...)
}

// Log prints a high level summary of the memory
func (m BackendMemory) Log(level slog.Level) {
	var total uint64

	for _, gpu := range m.GPUs {
		if sum := sumMemory(gpu.Weights); sum > 0 {
			slog.Log(context.TODO(), level, "model weights", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := m.InputWeights + sumMemory(m.CPU.Weights); sum > 0 {
		slog.Log(context.TODO(), level, "model weights", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	for _, gpu := range m.GPUs {
		if sum := sumMemory(gpu.Cache); sum > 0 {
			slog.Log(context.TODO(), level, "kv cache", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := sumMemory(m.CPU.Cache); sum > 0 {
		slog.Log(context.TODO(), level, "kv cache", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	for _, gpu := range m.GPUs {
		if sum := gpu.Graph; sum > 0 {
			slog.Log(context.TODO(), level, "compute graph", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := m.CPU.Graph; sum > 0 {
		slog.Log(context.TODO(), level, "compute graph", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	if total > 0 {
		slog.Log(context.TODO(), level, "total memory", "size", format.HumanBytes2(total))
	}
}
