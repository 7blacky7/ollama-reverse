// device_info.go
// Dieses Modul enthaelt die DeviceInfo-Strukturen und zugehoerige Funktionen
// fuer Geraete-Erkennung, Vergleich und Sortierung.
// Urspruenglich aus device.go extrahiert.

package ml

import (
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

type DeviceInfo struct {
	DeviceID

	// Name is the name of the device as labeled by the backend. It
	// may not be persistent across instances of the runner.
	Name string `json:"name"`

	// Description is the longer user-friendly identification of the device
	Description string `json:"description"`

	// FilterID is populated with the unfiltered device ID if a numeric ID is used
	// so the device can be included.
	FilterID string `json:"filter_id,omitempty"`

	// Integrated is set true for integrated GPUs, false for Discrete GPUs
	Integrated bool `json:"integration,omitempty"`

	// PCIID is the bus, device and domain ID of the device for deduplication
	// when discovered by multiple backends
	PCIID string `json:"pci_id,omitempty"`

	// TotalMemory is the total amount of memory the device can use for loading models
	TotalMemory uint64 `json:"total_memory"`

	// FreeMemory is the amount of memory currently available on the device for loading models
	FreeMemory uint64 `json:"free_memory,omitempty"`

	// ComputeMajor is the major version of capabilities of the device
	// if unsupported by the backend, -1 will be returned
	ComputeMajor int

	// ComputeMinor is the minor version of capabilities of the device
	// if unsupported by the backend, -1 will be returned
	ComputeMinor int

	// Driver Information
	DriverMajor int `json:"driver_major,omitempty"`
	DriverMinor int `json:"driver_minor,omitempty"`

	// Where backends were loaded from
	LibraryPath []string
}

type SystemInfo struct {
	// ThreadCount is the optimal number of threads to use for inference
	ThreadCount int `json:"threads,omitempty"`

	// TotalMemory is the total amount of system memory
	TotalMemory uint64 `json:"total_memory,omitempty"`

	// FreeMemory is the amount of memory currently available on the system for loading models
	FreeMemory uint64 `json:"free_memory,omitempty"`

	// FreeSwap is the amount of system swap space reported as available
	FreeSwap uint64 `json:"free_swap,omitempty"`
}

func (d DeviceInfo) Compute() string {
	// AMD gfx is encoded into the major minor in hex form
	if strings.EqualFold(d.Library, "ROCm") {
		return fmt.Sprintf("gfx%x%02x", d.ComputeMajor, d.ComputeMinor)
	}
	return strconv.Itoa(d.ComputeMajor) + "." + strconv.Itoa(d.ComputeMinor)
}

func (d DeviceInfo) Driver() string {
	return strconv.Itoa(d.DriverMajor) + "." + strconv.Itoa(d.DriverMinor)
}

// MinimumMemory reports the amount of memory that should be set aside
// on the device for overhead (e.g. VRAM consumed by context structures independent
// of model allocations)
func (d DeviceInfo) MinimumMemory() uint64 {
	if d.Library == "Metal" {
		return 512 * format.MebiByte
	}
	return 457 * format.MebiByte
}

// Sort by Free Space.
// iGPUs are reported first, thus Reverse() yields the largest discrete GPU first
type ByFreeMemory []DeviceInfo

func (a ByFreeMemory) Len() int      { return len(a) }
func (a ByFreeMemory) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByFreeMemory) Less(i, j int) bool {
	if a[i].Integrated && !a[j].Integrated {
		return true
	} else if !a[i].Integrated && a[j].Integrated {
		return false
	}
	return a[i].FreeMemory < a[j].FreeMemory
}

// ByPerformance groups devices by similar speed
func ByPerformance(l []DeviceInfo) [][]DeviceInfo {
	resp := [][]DeviceInfo{}
	scores := []bool{}
	for _, info := range l {
		found := false
		requested := info.Integrated
		for i, score := range scores {
			if score == requested {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			scores = append(scores, requested)
			resp = append(resp, []DeviceInfo{info})
		}
	}
	return resp
}

func ByLibrary(l []DeviceInfo) [][]DeviceInfo {
	resp := [][]DeviceInfo{}
	libs := []string{}
	for _, info := range l {
		found := false
		requested := info.Library
		for i, lib := range libs {
			if lib == requested {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			libs = append(libs, requested)
			resp = append(resp, []DeviceInfo{info})
		}
	}
	return resp
}

func LibraryPaths(l []DeviceInfo) []string {
	gpuLibs := []string{LibOllamaPath}
	for _, gpu := range l {
		for _, dir := range gpu.LibraryPath {
			needed := true
			for _, existing := range gpuLibs {
				if dir == existing {
					needed = false
					break
				}
			}
			if needed {
				gpuLibs = append(gpuLibs, dir)
			}
		}
	}
	return gpuLibs
}

type DeviceComparison int

const (
	UniqueDevice      DeviceComparison = iota
	SameBackendDevice                  // The device is the same, and the library/backend is the same
	DuplicateDevice                    // The same physical device but different library/backend (overlapping device)
)

func (a DeviceInfo) Compare(b DeviceInfo) DeviceComparison {
	if a.PCIID != b.PCIID {
		return UniqueDevice
	}
	// If PCIID is empty, we have to use ID + library for uniqueness
	if a.PCIID == "" && a.DeviceID != b.DeviceID {
		return UniqueDevice
	}
	if a.Library == b.Library {
		return SameBackendDevice
	}
	return DuplicateDevice
}

// For a SameBackendDevice, return true if b is better than a
// e.g. newer GPU library version
func (a DeviceInfo) IsBetter(b DeviceInfo) bool {
	aLib := a.LibraryPath[len(a.LibraryPath)-1]
	bLib := b.LibraryPath[len(b.LibraryPath)-1]
	if aLib == bLib {
		return false
	}
	aLibSplit := strings.SplitN(aLib, "_", 2)
	bLibSplit := strings.SplitN(bLib, "_", 2)
	if len(aLibSplit) < 2 || len(bLibSplit) < 2 {
		return false
	}
	if aLibSplit[0] != bLibSplit[0] {
		slog.Debug("unexpected libraries", "a", aLib, "b", bLib)
		return false
	}
	if aLibSplit[1] == bLibSplit[1] {
		return false
	}
	cmp := []string{aLibSplit[1], bLibSplit[1]}
	sort.Sort(sort.Reverse(sort.StringSlice(cmp)))
	return cmp[0] == bLibSplit[1]
}
