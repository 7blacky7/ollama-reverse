// MODUL: cuda
// ZWECK: CUDA Backend Detection und Device Management
// INPUT: Keine (Hardware-Abfrage)
// OUTPUT: DeviceInfo fuer CUDA-Geraete
// NEBENEFFEKTE: CGO-Aufrufe zur CUDA Runtime
// ABHAENGIGKEITEN: backend_detect.cpp (CGO)
// HINWEISE: Build-Tag "cuda" fuer bedingte Kompilierung

//go:build cuda

package backend

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbackend_detect

#include "backend_detect.h"
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"
)

// ============================================================================
// CUDADetector - CUDA Backend Erkennung
// ============================================================================

// CUDADetector implementiert Detector fuer CUDA-Backends.
type CUDADetector struct {
	available bool
	devices   []DeviceInfo
	checked   bool
}

// NewCUDADetector erstellt einen neuen CUDA-Detektor.
func NewCUDADetector() *CUDADetector {
	return &CUDADetector{}
}

// ============================================================================
// Detector Interface Implementierung
// ============================================================================

// Detect prueft ob CUDA verfuegbar ist.
func (d *CUDADetector) Detect() bool {
	if d.checked {
		return d.available
	}

	d.available = bool(C.backend_cuda_available())
	d.checked = true
	return d.available
}

// GetDevices gibt alle CUDA-Geraete zurueck.
func (d *CUDADetector) GetDevices() []DeviceInfo {
	if !d.Detect() {
		return nil
	}

	if d.devices != nil {
		return d.devices
	}

	// Anzahl der Geraete abfragen
	count := int(C.backend_cuda_device_count())
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
func (d *CUDADetector) Backend() Backend {
	return BackendCUDA
}

// ============================================================================
// Hilfsfunktionen
// ============================================================================

// queryDevice fragt Informationen fuer ein CUDA-Geraet ab.
func (d *CUDADetector) queryDevice(deviceID int) DeviceInfo {
	var cInfo C.backend_device_info
	C.backend_cuda_get_device(C.int(deviceID), &cInfo)

	return DeviceInfo{
		Backend:     BackendCUDA,
		DeviceID:    deviceID,
		DeviceName:  C.GoString(&cInfo.name[0]),
		MemoryTotal: uint64(cInfo.memory_total),
		MemoryFree:  uint64(cInfo.memory_free),
		ComputeCap:  formatComputeCap(int(cInfo.compute_major), int(cInfo.compute_minor)),
		IsDefault:   deviceID == 0,
	}
}

// formatComputeCap formatiert Compute Capability als String.
func formatComputeCap(major, minor int) string {
	// Einfache String-Konvertierung ohne fmt
	return intToStr(major) + "." + intToStr(minor)
}

// intToStr konvertiert int zu string (ohne fmt-Package).
func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

// ============================================================================
// Init - Registriere CUDA Detector beim Package-Load
// ============================================================================

func init() {
	RegisterDetector(BackendCUDA, NewCUDADetector())
}

// ============================================================================
// Memory Management Funktionen
// ============================================================================

// CUDAMemoryInfo gibt Speicherinfo fuer ein Geraet zurueck.
type CUDAMemoryInfo struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// GetMemoryInfo fragt aktuellen Speicherstand ab.
func GetCUDAMemoryInfo(deviceID int) CUDAMemoryInfo {
	var cInfo C.backend_device_info
	C.backend_cuda_get_device(C.int(deviceID), &cInfo)

	total := uint64(cInfo.memory_total)
	free := uint64(cInfo.memory_free)

	return CUDAMemoryInfo{
		Total: total,
		Free:  free,
		Used:  total - free,
	}
}

// SetDevice setzt das aktive CUDA-Geraet.
func SetCUDADevice(deviceID int) error {
	result := C.backend_cuda_set_device(C.int(deviceID))
	if result != 0 {
		return &BackendError{
			Backend: BackendCUDA,
			Op:      "set_device",
			Code:    int(result),
		}
	}
	return nil
}

// ============================================================================
// Fehlertypen
// ============================================================================

// BackendError repraesentiert einen Backend-spezifischen Fehler.
type BackendError struct {
	Backend Backend
	Op      string
	Code    int
}

// Error implementiert error Interface.
func (e *BackendError) Error() string {
	return string(e.Backend) + ": " + e.Op + " failed"
}

// Sicherstellen dass unsafe importiert wird (fuer CGO)
var _ = unsafe.Pointer(nil)
