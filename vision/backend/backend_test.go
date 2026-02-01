// MODUL: backend_test
// ZWECK: Unit-Tests fuer Backend-Detection und -Auswahl
// INPUT: Keine
// OUTPUT: Test-Ergebnisse
// NEBENEFFEKTE: Keine (nur Abfragen)
// ABHAENGIGKEITEN: testing (stdlib), backend.go
// HINWEISE: Tests laufen auf jeder Plattform (CPU immer verfuegbar)

package backend

import (
	"testing"
)

// ============================================================================
// Backend-Typ Tests
// ============================================================================

// TestBackendConstants prueft die Backend-Konstanten.
func TestBackendConstants(t *testing.T) {
	if BackendCPU != "cpu" {
		t.Errorf("BackendCPU: erwartet 'cpu', bekommen '%s'", BackendCPU)
	}
	if BackendCUDA != "cuda" {
		t.Errorf("BackendCUDA: erwartet 'cuda', bekommen '%s'", BackendCUDA)
	}
	if BackendMetal != "metal" {
		t.Errorf("BackendMetal: erwartet 'metal', bekommen '%s'", BackendMetal)
	}
}

// TestDefaultPriority prueft die Standard-Prioritaet.
func TestDefaultPriority(t *testing.T) {
	prio := DefaultPriority()

	if len(prio) != 3 {
		t.Fatalf("DefaultPriority: erwartet 3 Elemente, bekommen %d", len(prio))
	}
	if prio[0] != BackendCUDA {
		t.Errorf("DefaultPriority[0]: erwartet CUDA, bekommen %s", prio[0])
	}
	if prio[1] != BackendMetal {
		t.Errorf("DefaultPriority[1]: erwartet Metal, bekommen %s", prio[1])
	}
	if prio[2] != BackendCPU {
		t.Errorf("DefaultPriority[2]: erwartet CPU, bekommen %s", prio[2])
	}
}

// ============================================================================
// Detection Tests
// ============================================================================

// TestDetectBackends prueft die Backend-Erkennung.
func TestDetectBackends(t *testing.T) {
	backends := DetectBackends()

	// CPU muss immer dabei sein
	hasCPU := false
	for _, b := range backends {
		if b == BackendCPU {
			hasCPU = true
			break
		}
	}

	if !hasCPU {
		t.Error("DetectBackends: CPU sollte immer verfuegbar sein")
	}
}

// TestIsBackendAvailable prueft die Verfuegbarkeitspruefung.
func TestIsBackendAvailable(t *testing.T) {
	// CPU muss immer verfuegbar sein
	if !IsBackendAvailable(BackendCPU) {
		t.Error("IsBackendAvailable(CPU): sollte true sein")
	}
}

// TestSelectBestBackend prueft die automatische Backend-Auswahl.
func TestSelectBestBackend(t *testing.T) {
	best := SelectBestBackend()

	// Best muss einer der gueltigen Backends sein
	valid := best == BackendCPU || best == BackendCUDA || best == BackendMetal
	if !valid {
		t.Errorf("SelectBestBackend: ungueltiges Backend '%s'", best)
	}
}

// ============================================================================
// Device Info Tests
// ============================================================================

// TestGetDevices prueft die Geraete-Abfrage.
func TestGetDevices(t *testing.T) {
	devices := GetDevices()

	// Mindestens CPU muss vorhanden sein
	if len(devices) == 0 {
		t.Error("GetDevices: mindestens ein Geraet erwartet")
	}

	// Erstes Geraet sollte CPU sein (oder GPU wenn verfuegbar)
	for _, dev := range devices {
		if dev.Backend == BackendCPU {
			if dev.DeviceName != "CPU" {
				t.Errorf("CPU DeviceName: erwartet 'CPU', bekommen '%s'", dev.DeviceName)
			}
			break
		}
	}
}

// TestCPUDeviceInfo prueft CPU-Geraete-Informationen.
func TestCPUDeviceInfo(t *testing.T) {
	info := cpuDeviceInfo()

	if info.Backend != BackendCPU {
		t.Errorf("cpuDeviceInfo Backend: erwartet CPU, bekommen %s", info.Backend)
	}
	if info.DeviceID != 0 {
		t.Errorf("cpuDeviceInfo DeviceID: erwartet 0, bekommen %d", info.DeviceID)
	}
	if !info.IsDefault {
		t.Error("cpuDeviceInfo IsDefault: sollte true sein")
	}
}

// ============================================================================
// Priority Selection Tests
// ============================================================================

// TestSelectBestBackendWithPriority prueft benutzerdefinierte Prioritaet.
func TestSelectBestBackendWithPriority(t *testing.T) {
	// CPU-zuerst Prioritaet
	cpuFirst := SelectionPriority{BackendCPU, BackendCUDA, BackendMetal}
	best := SelectBestBackendWithPriority(cpuFirst)

	// Mit CPU-zuerst sollte CPU gewaehlt werden
	if best != BackendCPU {
		t.Errorf("SelectBestBackendWithPriority(CPU first): erwartet CPU, bekommen %s", best)
	}
}

// TestSelectBestBackendEmptyPriority prueft leere Prioritaet.
func TestSelectBestBackendEmptyPriority(t *testing.T) {
	empty := SelectionPriority{}
	best := SelectBestBackendWithPriority(empty)

	// Bei leerer Prioritaet sollte CPU als Fallback gewaehlt werden
	if best != BackendCPU {
		t.Errorf("SelectBestBackendWithPriority(empty): erwartet CPU, bekommen %s", best)
	}
}

// ============================================================================
// Registry Tests
// ============================================================================

// TestRegisterDetector prueft die Detektor-Registrierung.
func TestRegisterDetector(t *testing.T) {
	// Dummy-Detektor fuer Test
	dummy := &testDetector{available: true}

	// Registrieren unter neuem Namen (um existierende nicht zu ueberschreiben)
	RegisterDetector("test_backend", dummy)

	// Abfrage sollte den Detektor finden (implizit ueber registeredDetectors)
	if d, ok := registeredDetectors["test_backend"]; !ok || d != dummy {
		t.Error("RegisterDetector: Detektor nicht korrekt registriert")
	}

	// Cleanup
	delete(registeredDetectors, "test_backend")
}

// testDetector ist ein Dummy-Detektor fuer Tests.
type testDetector struct {
	available bool
}

func (d *testDetector) Detect() bool          { return d.available }
func (d *testDetector) GetDevices() []DeviceInfo { return nil }
func (d *testDetector) Backend() Backend      { return "test" }
