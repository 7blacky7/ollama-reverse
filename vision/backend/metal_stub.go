// MODUL: metal_stub
// ZWECK: Stub-Implementierung wenn Metal nicht verfuegbar
// INPUT: Keine
// OUTPUT: Leere DeviceInfo-Liste, false fuer Detect
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: backend.go
// HINWEISE: Wird kompiliert wenn NICHT darwin ODER Build-Tag "metal" NICHT gesetzt

//go:build !darwin || !metal

package backend

// ============================================================================
// MetalDetector Stub - Keine Metal Unterstuetzung
// ============================================================================

// MetalDetector Stub fuer Builds ohne Metal.
type MetalDetector struct{}

// NewMetalDetector erstellt Stub-Detektor.
func NewMetalDetector() *MetalDetector {
	return &MetalDetector{}
}

// Detect gibt immer false zurueck.
func (d *MetalDetector) Detect() bool {
	return false
}

// GetDevices gibt leere Liste zurueck.
func (d *MetalDetector) GetDevices() []DeviceInfo {
	return nil
}

// Backend gibt BackendMetal zurueck.
func (d *MetalDetector) Backend() Backend {
	return BackendMetal
}

// ============================================================================
// Stub-Funktionen fuer Metal Memory Management
// ============================================================================

// MetalMemoryInfo Stub-Struktur.
type MetalMemoryInfo struct {
	Total          uint64
	Recommended    uint64
	CurrentUsed    uint64
	HasUnifiedMem  bool
}

// GetMetalMemoryInfo gibt leere Info zurueck.
func GetMetalMemoryInfo(deviceID int) MetalMemoryInfo {
	return MetalMemoryInfo{}
}

// IsAppleSilicon gibt false zurueck auf nicht-Apple Systemen.
func IsAppleSilicon() bool {
	return false
}
