// MODUL: cuda_stub
// ZWECK: Stub-Implementierung wenn CUDA nicht verfuegbar
// INPUT: Keine
// OUTPUT: Leere DeviceInfo-Liste, false fuer Detect
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: backend.go
// HINWEISE: Wird kompiliert wenn Build-Tag "cuda" NICHT gesetzt

//go:build !cuda

package backend

// ============================================================================
// CUDADetector Stub - Keine CUDA Unterstuetzung
// ============================================================================

// CUDADetector Stub fuer Builds ohne CUDA.
type CUDADetector struct{}

// NewCUDADetector erstellt Stub-Detektor.
func NewCUDADetector() *CUDADetector {
	return &CUDADetector{}
}

// Detect gibt immer false zurueck.
func (d *CUDADetector) Detect() bool {
	return false
}

// GetDevices gibt leere Liste zurueck.
func (d *CUDADetector) GetDevices() []DeviceInfo {
	return nil
}

// Backend gibt BackendCUDA zurueck.
func (d *CUDADetector) Backend() Backend {
	return BackendCUDA
}

// ============================================================================
// Stub-Funktionen fuer CUDA Memory Management
// ============================================================================

// CUDAMemoryInfo Stub-Struktur.
type CUDAMemoryInfo struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// GetCUDAMemoryInfo gibt leere Info zurueck.
func GetCUDAMemoryInfo(deviceID int) CUDAMemoryInfo {
	return CUDAMemoryInfo{}
}

// SetCUDADevice gibt immer Fehler zurueck.
func SetCUDADevice(deviceID int) error {
	return &BackendError{
		Backend: BackendCUDA,
		Op:      "set_device",
		Code:    -1,
	}
}

// ============================================================================
// Fehlertypen (gleich wie in cuda.go)
// ============================================================================

// BackendError repraesentiert einen Backend-spezifischen Fehler.
type BackendError struct {
	Backend Backend
	Op      string
	Code    int
}

// Error implementiert error Interface.
func (e *BackendError) Error() string {
	return string(e.Backend) + ": " + e.Op + " not available"
}
