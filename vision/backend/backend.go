// MODUL: backend
// ZWECK: Abstraktion fuer Compute-Backends (CPU/CUDA/Metal)
// INPUT: Keine (reine Datenstrukturen und Detection)
// OUTPUT: Backend-Typ, DeviceInfo, Verfuegbarkeit
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: Keine externen (nur stdlib)
// HINWEISE: Platform-spezifische Implementierung in cuda.go/metal.go

package backend

// ============================================================================
// Backend-Typ Definition
// ============================================================================

// Backend repraesentiert ein verfuegbares Compute-Backend.
type Backend string

// Verfuegbare Backend-Typen
const (
	BackendCPU   Backend = "cpu"
	BackendCUDA  Backend = "cuda"
	BackendMetal Backend = "metal"
)

// ============================================================================
// DeviceInfo - Hardware-Informationen
// ============================================================================

// DeviceInfo enthaelt Informationen ueber ein verfuegbares Compute-Geraet.
type DeviceInfo struct {
	Backend     Backend // Backend-Typ (cpu, cuda, metal)
	DeviceID    int     // Geraete-Index (0 fuer CPU, GPU-Index sonst)
	DeviceName  string  // Lesbarer Geraetename
	MemoryTotal uint64  // Gesamter Speicher in Bytes
	MemoryFree  uint64  // Freier Speicher in Bytes
	ComputeCap  string  // Compute Capability (z.B. "8.6" fuer CUDA)
	IsDefault   bool    // Ob dies das Standard-Geraet ist
}

// ============================================================================
// Backend-Selection Optionen
// ============================================================================

// SelectionPriority definiert Praeferenzreihenfolge fuer Backend-Auswahl.
type SelectionPriority []Backend

// DefaultPriority gibt die Standard-Praeferenzreihenfolge zurueck.
func DefaultPriority() SelectionPriority {
	return SelectionPriority{BackendCUDA, BackendMetal, BackendCPU}
}

// ============================================================================
// Detection Interface
// ============================================================================

// Detector ist das Interface fuer Backend-Erkennung.
// Implementierungen: CPUDetector, CUDADetector, MetalDetector
type Detector interface {
	// Detect prueft ob das Backend verfuegbar ist
	Detect() bool

	// GetDevices gibt alle verfuegbaren Geraete zurueck
	GetDevices() []DeviceInfo

	// Backend gibt den Backend-Typ zurueck
	Backend() Backend
}

// ============================================================================
// Globale Detection-Funktionen
// ============================================================================

// registeredDetectors haelt alle registrierten Backend-Detektoren.
var registeredDetectors = make(map[Backend]Detector)

// RegisterDetector registriert einen Detektor fuer ein Backend.
func RegisterDetector(b Backend, d Detector) {
	registeredDetectors[b] = d
}

// DetectBackends erkennt alle verfuegbaren Backends.
func DetectBackends() []Backend {
	var available []Backend

	// CPU ist immer verfuegbar
	available = append(available, BackendCPU)

	// Pruefe CUDA
	if d, ok := registeredDetectors[BackendCUDA]; ok && d.Detect() {
		available = append(available, BackendCUDA)
	}

	// Pruefe Metal
	if d, ok := registeredDetectors[BackendMetal]; ok && d.Detect() {
		available = append(available, BackendMetal)
	}

	return available
}

// GetDevices gibt alle verfuegbaren Geraete zurueck.
func GetDevices() []DeviceInfo {
	var devices []DeviceInfo

	// CPU-Geraet hinzufuegen
	devices = append(devices, cpuDeviceInfo())

	// CUDA-Geraete
	if d, ok := registeredDetectors[BackendCUDA]; ok && d.Detect() {
		devices = append(devices, d.GetDevices()...)
	}

	// Metal-Geraete
	if d, ok := registeredDetectors[BackendMetal]; ok && d.Detect() {
		devices = append(devices, d.GetDevices()...)
	}

	return devices
}

// SelectBestBackend waehlt das optimale Backend basierend auf Prioritaet.
func SelectBestBackend() Backend {
	return SelectBestBackendWithPriority(DefaultPriority())
}

// SelectBestBackendWithPriority waehlt Backend nach gegebener Prioritaet.
func SelectBestBackendWithPriority(priority SelectionPriority) Backend {
	available := DetectBackends()
	availableSet := make(map[Backend]bool)
	for _, b := range available {
		availableSet[b] = true
	}

	for _, preferred := range priority {
		if availableSet[preferred] {
			return preferred
		}
	}

	return BackendCPU
}

// IsBackendAvailable prueft ob ein bestimmtes Backend verfuegbar ist.
func IsBackendAvailable(b Backend) bool {
	if b == BackendCPU {
		return true
	}
	if d, ok := registeredDetectors[b]; ok {
		return d.Detect()
	}
	return false
}

// ============================================================================
// CPU Device Info (immer verfuegbar)
// ============================================================================

// cpuDeviceInfo gibt Informationen ueber CPU zurueck.
func cpuDeviceInfo() DeviceInfo {
	return DeviceInfo{
		Backend:    BackendCPU,
		DeviceID:   0,
		DeviceName: "CPU",
		IsDefault:  true,
	}
}
