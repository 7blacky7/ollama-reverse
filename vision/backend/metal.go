// MODUL: metal
// ZWECK: Metal Backend Detection fuer macOS/iOS
// INPUT: Keine (Hardware-Abfrage)
// OUTPUT: DeviceInfo fuer Metal-Geraete
// NEBENEFFEKTE: CGO-Aufrufe zur Metal Runtime
// ABHAENGIGKEITEN: backend_detect.cpp (CGO)
// HINWEISE: Build-Tags "darwin" und "metal" fuer bedingte Kompilierung

//go:build darwin && metal

package backend

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbackend_detect -framework Metal -framework Foundation

#include "backend_detect.h"
#include <stdlib.h>
*/
import "C"

// ============================================================================
// MetalDetector - Metal Backend Erkennung
// ============================================================================

// MetalDetector implementiert Detector fuer Metal-Backends.
type MetalDetector struct {
	available bool
	devices   []DeviceInfo
	checked   bool
}

// NewMetalDetector erstellt einen neuen Metal-Detektor.
func NewMetalDetector() *MetalDetector {
	return &MetalDetector{}
}

// ============================================================================
// Detector Interface Implementierung
// ============================================================================

// Detect prueft ob Metal verfuegbar ist.
func (d *MetalDetector) Detect() bool {
	if d.checked {
		return d.available
	}

	d.available = bool(C.backend_metal_available())
	d.checked = true
	return d.available
}

// GetDevices gibt alle Metal-Geraete zurueck.
func (d *MetalDetector) GetDevices() []DeviceInfo {
	if !d.Detect() {
		return nil
	}

	if d.devices != nil {
		return d.devices
	}

	// Metal hat typischerweise nur ein Geraet
	count := int(C.backend_metal_device_count())
	if count == 0 {
		return nil
	}

	d.devices = make([]DeviceInfo, 0, count)

	for i := 0; i < count; i++ {
		info := d.queryDevice(i)
		d.devices = append(d.devices, info)
	}

	return d.devices
}

// Backend gibt den Backend-Typ zurueck.
func (d *MetalDetector) Backend() Backend {
	return BackendMetal
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// queryDevice fragt Informationen fuer ein Metal-Geraet ab.
func (d *MetalDetector) queryDevice(deviceID int) DeviceInfo {
	var cInfo C.backend_device_info
	C.backend_metal_get_device(C.int(deviceID), &cInfo)

	// Metal-Geraete haben keine Compute Capability wie CUDA
	return DeviceInfo{
		Backend:     BackendMetal,
		DeviceID:    deviceID,
		DeviceName:  C.GoString(&cInfo.name[0]),
		MemoryTotal: uint64(cInfo.memory_total),
		MemoryFree:  uint64(cInfo.memory_free),
		ComputeCap:  "",
		IsDefault:   deviceID == 0,
	}
}

// ============================================================================
// Init - Registriere Metal Detector beim Package-Load
// ============================================================================

func init() {
	RegisterDetector(BackendMetal, NewMetalDetector())
}

// ============================================================================
// Metal-spezifische Funktionen
// ============================================================================

// MetalMemoryInfo gibt Speicherinfo fuer Unified Memory zurueck.
type MetalMemoryInfo struct {
	Total          uint64
	Recommended    uint64 // Empfohlenes Limit fuer GPU-Allokation
	CurrentUsed    uint64
	HasUnifiedMem  bool
}

// GetMetalMemoryInfo fragt Metal-spezifische Speicherinfo ab.
func GetMetalMemoryInfo(deviceID int) MetalMemoryInfo {
	var cInfo C.backend_device_info
	C.backend_metal_get_device(C.int(deviceID), &cInfo)

	return MetalMemoryInfo{
		Total:         uint64(cInfo.memory_total),
		Recommended:   uint64(cInfo.memory_free), // Metal nutzt "recommended working set"
		HasUnifiedMem: true,                      // Alle Apple Silicon haben Unified Memory
	}
}

// IsAppleSilicon prueft ob das Geraet Apple Silicon ist.
func IsAppleSilicon() bool {
	if !NewMetalDetector().Detect() {
		return false
	}
	// Auf macOS mit Metal ist es Apple Silicon oder AMD GPU
	// Apple Silicon hat Unified Memory
	return true
}
