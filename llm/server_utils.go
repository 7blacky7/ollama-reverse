// Package llm - Utility-Funktionen für Server
//
// Hilfsfunktionen für Model-Loading:
// - projectorMemoryRequirements: Berechnet Projektor-Speicherbedarf
// - uniqueDeviceIDs: Extrahiert eindeutige Device IDs
package llm

import (
	"os"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// projectorMemoryRequirements berechnet Speicherbedarf eines Projektors
func projectorMemoryRequirements(filename string) (weights uint64) {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	ggml, err := ggml.Decode(file, 1024)
	if err != nil {
		return 0
	}

	for _, layer := range ggml.Tensors().GroupLayers() {
		weights += layer.Size()
	}

	return weights
}

// uniqueDeviceIDs extrahiert eindeutige Device IDs aus GPULayersList
func uniqueDeviceIDs(gpuLayers ml.GPULayersList) []ml.DeviceID {
	devices := []ml.DeviceID{}
	for _, layer := range gpuLayers {
		isNew := true
		for _, ID := range devices {
			if layer.DeviceID == ID {
				isNew = false
				break
			}
		}
		if isNew {
			devices = append(devices, layer.DeviceID)
		}
	}
	return devices
}
