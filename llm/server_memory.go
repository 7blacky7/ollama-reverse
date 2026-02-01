// Package llm - Memory Layout Management
//
// Funktionen zur Speicherverwaltung beim Model-Laden:
// - initializeMemoryLayout: Initialisiert Speicherstrukturen
// - assignLayersToCPU: Weist Layer der CPU zu
// - handleProjector: Verwaltet Projektor-Speicher
// - adjustGraphSizes: Passt Graph-Größen an
// - iterateLayouts: Iteriert über Layout-Konfigurationen
// - finalizeMemoryLayout: Finalisiert Speicher-Layout
// - configureMmap: Konfiguriert Memory-Mapping
package llm

import (
	"fmt"
	"log/slog"
	"runtime"

	"github.com/ollama/ollama/ml"
)

func (s *llmServer) initializeMemoryLayout(gpus []ml.DeviceInfo) {
	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Name:    "CPU",
			Weights: make([]uint64, s.totalLayers),
			Cache:   make([]uint64, s.totalLayers),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].Name = gpus[i].Name
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, s.totalLayers)
		s.mem.GPUs[i].Cache = make([]uint64, s.totalLayers)
	}
}

func (s *llamaServer) assignLayersToCPU(kv []uint64, gpus []ml.DeviceInfo) {
	layers := s.ggml.Tensors().GroupLayers()

	if blk0, ok := layers["blk.0"]; ok {
		buffer := blk0.Size() + kv[0]
		for i := range gpus {
			if gpus[i].FreeMemory > buffer {
				gpus[i].FreeMemory -= buffer
			} else {
				gpus[i].FreeMemory = 0
			}
		}
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	for i := range s.ggml.KV().BlockCount() {
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			s.mem.CPU.Weights[i] = blk.Size()
			s.mem.CPU.Cache[i] += kv[i]
		}
	}

	var outputWeights uint64
	if layer, ok := layers["output_norm"]; ok {
		outputWeights += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		outputWeights += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		outputWeights += layer.Size()
	}
	s.mem.CPU.Weights[s.totalLayers-1] = outputWeights
}

func (s *llamaServer) handleProjector(gpus []ml.DeviceInfo) int {
	if len(gpus) == 0 {
		return -1
	}

	var projectorWeights uint64
	for _, projector := range s.loadRequest.LoraPath {
		projectorWeights += projectorMemoryRequirements(projector)
	}

	projectorGPU := findProjectorGPU(gpus)
	if gpus[projectorGPU].FreeMemory > projectorWeights {
		gpus[projectorGPU].FreeMemory -= projectorWeights
	} else {
		gpus[projectorGPU].FreeMemory = 0
	}

	return projectorGPU
}

func findProjectorGPU(gpus []ml.DeviceInfo) int {
	firstIntegrated := -1
	for i := range gpus {
		if !gpus[i].Integrated {
			return i
		}
		if firstIntegrated == -1 {
			firstIntegrated = i
		}
	}
	return firstIntegrated
}

func (s *llamaServer) adjustGraphSizes(kv []uint64, partial, full uint64, gpus []ml.DeviceInfo) (uint64, uint64) {
	var kvTotal uint64
	for _, kvLayer := range kv {
		kvTotal += kvLayer
	}

	if partial == 0 {
		headsKV := s.ggml.KV().HeadCountKVMin()
		if headsKV == 0 {
			headsKV = 1
		}
		gqa := s.ggml.KV().HeadCountMax() / headsKV
		partial = gqa * kvTotal / 6
	}
	if full == 0 {
		full = partial
	}
	if len(gpus) > 0 && gpus[0].Library == "Metal" {
		partial = full
	}

	return partial, full
}

func (s *llamaServer) iterateLayouts(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool, graphPartial, graphFull uint64) (ml.GPULayersList, error) {
	var gpuLayers ml.GPULayersList

	for {
		prevGPULayers := gpuLayers
		var err error
		gpuLayers, err = s.createLayout(systemInfo, gpus, s.mem, requireFull, 0)
		if err != nil {
			return nil, err
		}

		if len(gpuLayers) > len(prevGPULayers) {
			for _, gl := range gpuLayers {
				for i := range s.mem.GPUs {
					if gl.DeviceID == s.mem.GPUs[i].DeviceID {
						s.mem.GPUs[i].Graph = max(graphPartial, graphFull)
						break
					}
				}
			}
		} else {
			break
		}
	}

	return gpuLayers, nil
}

func (s *llamaServer) finalizeMemoryLayout(gpuLayers ml.GPULayersList, projectorGPU int, graphPartial, graphFull uint64) {
	graphSize := graphFull
	if gpuLayers.Sum() < int(s.totalLayers) {
		graphSize = graphPartial
	}

	for _, gl := range gpuLayers {
		for i := range s.mem.GPUs {
			if gl.DeviceID == s.mem.GPUs[i].DeviceID {
				for _, l := range gl.Layers {
					s.mem.GPUs[i].Weights[l] = s.mem.CPU.Weights[l]
					s.mem.GPUs[i].Cache[l] = s.mem.CPU.Cache[l]
					s.mem.CPU.Weights[l] = 0
					s.mem.CPU.Cache[l] = 0
				}
				s.mem.GPUs[i].Graph = graphSize
				break
			}
		}
	}

	if projectorGPU > 0 && len(s.mem.GPUs) > projectorGPU && len(s.mem.GPUs[projectorGPU].Weights) > 0 {
		projWeights := uint64(0)
		for _, p := range s.loadRequest.LoraPath {
			projWeights += projectorMemoryRequirements(p)
		}
		s.mem.GPUs[projectorGPU].Weights[s.totalLayers-1] += projWeights
	}

	slog.Debug("memory", "estimate", s.mem)
	s.mem.Log(slog.LevelInfo)
}

func (s *llamaServer) configureMmap(gpus []ml.DeviceInfo, systemInfo ml.SystemInfo) {
	s.loadRequest.UseMmap = true

	for _, g := range gpus {
		if g.Library == "Metal" && uint64(s.options.NumGPU) > 0 && uint64(s.options.NumGPU) < s.totalLayers {
			s.options.UseMMap = new(bool)
			*s.options.UseMMap = false
		}
	}

	if shouldDisableMmap(gpus, systemInfo, s.options.UseMMap, s.TotalSize()) {
		s.loadRequest.UseMmap = false
	}
}

func shouldDisableMmap(gpus []ml.DeviceInfo, systemInfo ml.SystemInfo, useMMap *bool, totalSize uint64) bool {
	if useMMap != nil && !*useMMap {
		return true
	}
	if useMMap != nil {
		return false
	}
	if len(gpus) == 0 {
		return true
	}
	if runtime.GOOS == "windows" && gpus[0].Library == "CUDA" {
		return true
	}
	if runtime.GOOS == "linux" && systemInfo.FreeMemory < totalSize {
		return true
	}
	if gpus[0].Library == "Vulkan" {
		return true
	}
	return false
}
