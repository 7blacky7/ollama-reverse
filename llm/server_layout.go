// Package llm - Memory Layout Berechnung
//
// Algorithmen zur optimalen Verteilung von Model-Layern auf GPUs:
// - createLayout: Haupt-Layout-Funktion mit Speichervalidierung
// - buildLayout: Berechnet Layer-Zuweisung basierend auf freiem Speicher
// - verifyLayout: Prüft Limits (partial offloading, System-Speicher)
// - assignLayers: Packt Layer auf minimale GPU-Anzahl
// - findBestFit/greedyFit: Optimierungs-Algorithmen
package llm

import (
	"fmt"
	"log/slog"
	"runtime"
	"sort"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// createLayout erstellt ein optimales Memory-Layout für GPUs
func (s *llmServer) createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, error) {
	if memory == nil {
		memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
			Weights: make([]uint64, s.totalLayers),
			Cache:   make([]uint64, s.totalLayers),
		}}
	}
	gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
	err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
	if err != nil {
		return nil, err
	}
	return gpuLayers, nil
}

// buildLayout berechnet die Layer-Zuweisung basierend auf verfügbarem Speicher
func (s *llmServer) buildLayout(systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, []uint64) {
	gpus := append(make([]ml.DeviceInfo, 0, len(systemGPUs)), systemGPUs...)
	sort.Sort(sort.Reverse(ml.ByFreeMemory(gpus)))

	layers := s.calculateLayerSizes(memory)
	gpuLayers := s.assignToGPUs(layers, gpus, memory, requireFull, backoff)

	return gpuLayers, layers
}

func (s *llmServer) calculateLayerSizes(memory *ml.BackendMemory) []uint64 {
	layers := make([]uint64, len(memory.CPU.Weights))

	for i := range layers {
		for j := range memory.GPUs {
			layers[i] += memory.GPUs[j].Weights[i]
			layers[i] += memory.GPUs[j].Cache[i]
		}
		layers[i] += memory.CPU.Weights[i]
		layers[i] += memory.CPU.Cache[i]
		logutil.Trace("layer to assign", "layer", i, "size", format.HumanBytes2(layers[i]))
	}

	return layers
}

func (s *llmServer) assignToGPUs(layers []uint64, gpus []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) ml.GPULayersList {
	gpuLayers := ml.GPULayersList{}

	for _, gl := range ml.ByLibrary(gpus) {
		lastUsedGPU := s.findLastUsedGPU(gl, memory)
		s.adjustGPUFreeMemory(gl, memory, backoff)

		libraryGpuLayers := assignLayers(layers, gl, requireFull, s.options.NumGPU, lastUsedGPU)
		if libraryGpuLayers.Sum() > gpuLayers.Sum() {
			gpuLayers = libraryGpuLayers
		}
	}

	return gpuLayers
}

func (s *llmServer) findLastUsedGPU(gpus []ml.DeviceInfo, memory *ml.BackendMemory) int {
	lastUsedGPU := 0
	for i := range gpus {
		for j := range memory.GPUs {
			if gpus[i].DeviceID == memory.GPUs[j].DeviceID && memory.GPUs[j].Graph != 0 {
				lastUsedGPU = i
			}
		}
	}
	return lastUsedGPU
}

func (s *llmServer) adjustGPUFreeMemory(gpus []ml.DeviceInfo, memory *ml.BackendMemory, backoff float32) {
	for i := range gpus {
		found := false
		for j := range memory.GPUs {
			if gpus[i].DeviceID == memory.GPUs[j].DeviceID {
				reserved := uint64(float32(gpus[i].FreeMemory)*backoff) +
					gpus[i].MinimumMemory() +
					envconfig.GpuOverhead() +
					memory.GPUs[j].Graph

				if gpus[i].FreeMemory > reserved {
					gpus[i].FreeMemory -= reserved
				} else {
					gpus[i].FreeMemory = 0
				}

				slog.Debug("available gpu", "id", gpus[i].ID, "library", gpus[i].Library,
					"available layer vram", format.HumanBytes2(gpus[i].FreeMemory),
					"backoff", fmt.Sprintf("%.2f", backoff),
					"minimum", format.HumanBytes2(gpus[i].MinimumMemory()),
					"overhead", format.HumanBytes2(envconfig.GpuOverhead()),
					"graph", format.HumanBytes2(memory.GPUs[j].Graph))

				found = true
				break
			}
		}
		if !found {
			gpus[i].FreeMemory = 0
		}
	}
}

// verifyLayout prüft ob das Layout die Limits einhält
func (s *llmServer) verifyLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, gpuLayers ml.GPULayersList, layers []uint64) error {
	cpuSize, vramSize := s.calculateMemoryUsage(memory, gpuLayers, layers)

	if err := s.checkFullRequirement(requireFull, systemGPUs, gpuLayers, layers, cpuSize, systemInfo); err != nil {
		return err
	}

	if err := s.checkSystemMemory(systemInfo, cpuSize, vramSize, systemGPUs, gpuLayers); err != nil {
		return err
	}

	if len(systemGPUs) > 0 && gpuLayers.Sum() == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
	}

	return nil
}

func (s *llmServer) calculateMemoryUsage(memory *ml.BackendMemory, gpuLayers ml.GPULayersList, layers []uint64) (cpuSize, vramSize uint64) {
	cpuSize = memory.InputWeights + memory.CPU.Graph

	for _, gl := range gpuLayers {
		for _, gpu := range memory.GPUs {
			if gl.DeviceID == gpu.DeviceID {
				vramSize += gpu.Graph
				break
			}
		}
	}

nextLayer:
	for i := range layers {
		for _, g := range gpuLayers {
			for _, gl := range g.Layers {
				if i == gl {
					vramSize += layers[i]
					continue nextLayer
				}
			}
		}
		cpuSize += layers[i]
	}

	return cpuSize, vramSize
}

func (s *llmServer) checkFullRequirement(requireFull bool, systemGPUs []ml.DeviceInfo, gpuLayers ml.GPULayersList, layers []uint64, cpuSize uint64, systemInfo ml.SystemInfo) error {
	if !requireFull {
		return nil
	}

	if len(systemGPUs) > 0 && gpuLayers.Sum() < len(layers) && (s.options.NumGPU < 0 || gpuLayers.Sum() < s.options.NumGPU) {
		slog.Info("model requires more gpu memory than is currently available, evicting a model to make space", "loaded layers", gpuLayers.Sum())
		return ErrLoadRequiredFull
	}

	if cpuSize > systemInfo.FreeMemory {
		slog.Info("model requires more system memory than is currently available, evicting a model to make space", "required", cpuSize, "free", systemInfo.FreeMemory)
		return fmt.Errorf("model requires more system memory than is currently available %w", ErrLoadRequiredFull)
	}

	return nil
}

func (s *llmServer) checkSystemMemory(systemInfo ml.SystemInfo, cpuSize, vramSize uint64, systemGPUs []ml.DeviceInfo, gpuLayers ml.GPULayersList) error {
	// Darwin hat dynamischen Swap, keine direkte Prüfung möglich
	if runtime.GOOS != "darwin" {
		available := systemInfo.FreeMemory + systemInfo.FreeSwap
		if cpuSize > available {
			slog.Warn("model request too large for system",
				"requested", format.HumanBytes2(cpuSize),
				"available", format.HumanBytes2(available),
				"total", format.HumanBytes2(systemInfo.TotalMemory),
				"free", format.HumanBytes2(systemInfo.FreeMemory),
				"swap", format.HumanBytes2(systemInfo.FreeSwap))
			return fmt.Errorf("model requires more system memory (%s) than is available (%s)",
				format.HumanBytes2(cpuSize), format.HumanBytes2(available))
		}
	} else {
		// Darwin: Deaktiviere partial offloading wenn Model > Total Memory
		if vramSize > systemInfo.TotalMemory {
			s.options.NumGPU = 0
		}
	}

	return nil
}

// assignLayers packt maximal viele Layer auf minimale GPU-Anzahl
func assignLayers(layers []uint64, gpus []ml.DeviceInfo, requireFull bool, requestedLayers int, lastUsedGPU int) (gpuLayers ml.GPULayersList) {
	// Bei manueller Überschreibung alle GPUs gleich behandeln
	if requestedLayers >= 0 || envconfig.SchedSpread() {
		for i := range gpus {
			gpus[i].Integrated = false
		}
	}

	// Zwei Durchläufe: mit und ohne Output-Layer
	for range 2 {
		requestedLayers = min(len(layers), requestedLayers)

		if !envconfig.SchedSpread() {
			for i := lastUsedGPU; i < len(gpus); i++ {
				forceRequest := i == len(gpus)-1 && !requireFull
				gpuLayers = findBestFit(layers, gpus[:i+1], requestedLayers, forceRequest)
				if gpuLayers.Sum() == len(layers) || gpuLayers.Sum() == requestedLayers {
					break
				}
			}
		} else {
			gpuLayers = findBestFit(layers, gpus, requestedLayers, !requireFull)
		}

		if gpuLayers.Sum() == len(layers) {
			return gpuLayers
		}

		layers = layers[:len(layers)-1]
	}

	return gpuLayers
}

// findBestFit sucht per Binary Search den kleinsten Kapazitätsfaktor
func findBestFit(layers []uint64, gpus []ml.DeviceInfo, requestedLayers int, forceRequest bool) (gpuLayers ml.GPULayersList) {
	for _, gl := range ml.ByPerformance(gpus) {
		var high float32 = 1
		var low float32 = 0

		// Bei forcierter Anforderung: quasi unbegrenzten VRAM annehmen
		if requestedLayers >= 0 && forceRequest {
			high = 1000
		}

		bestAssignments := greedyFit(layers, gl, high, requestedLayers)
		maxNumGPU := bestAssignments.Sum()

		// Binary Search für optimalen Kapazitätsfaktor
		for high-low > 1e-6 {
			mid := (low + high) / 2
			assignments := greedyFit(layers, gl, mid, requestedLayers)
			if assignments.Sum() == maxNumGPU {
				high = mid
				bestAssignments = assignments
			} else {
				low = mid
			}
		}

		layers = layers[:len(layers)-bestAssignments.Sum()]
		requestedLayers -= bestAssignments.Sum()
		gpuLayers = append(bestAssignments, gpuLayers...)
	}

	return gpuLayers
}

// greedyFit weist Layer inkrementell GPUs zu, bis Speicher aufgebraucht
func greedyFit(layers []uint64, gpus []ml.DeviceInfo, capacity float32, requestedLayers int) (gpuLayers ml.GPULayersList) {
	device := len(gpus) - 1
	gpuLayers = ml.GPULayersList{{DeviceID: gpus[device].DeviceID}}
	freeSpace := uint64(float32(gpus[device].FreeMemory) * capacity)

	for i := len(layers) - 1; i >= 0; i-- {
		if requestedLayers >= 0 && len(layers)-1-i >= requestedLayers {
			break
		}

		for {
			if layers[i] <= freeSpace {
				gpuLayers[0].Layers = append([]int{i}, gpuLayers[0].Layers...)
				freeSpace -= layers[i]
				break
			}

			device--
			if device < 0 {
				return gpuLayers
			}
			gpuLayers = append(ml.GPULayersList{{DeviceID: gpus[device].DeviceID}}, gpuLayers...)
			freeSpace = uint64(float32(gpus[device].FreeMemory) * capacity)
		}
	}

	return gpuLayers
}
