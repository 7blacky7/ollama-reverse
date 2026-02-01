// Modul: runner_refresh.go
// Beschreibung: GPU-Memory-Refresh-Logik nach dem Bootstrap.
// Aktualisiert VRAM-Informationen ueber bestehende Runner oder Bootstrap.

package discover

import (
	"context"
	"log/slog"
	"runtime"
	"time"

	"github.com/ollama/ollama/ml"
)

// refreshDevices aktualisiert die VRAM-Informationen der bekannten GPUs.
// Wird aufgerufen wenn bootstrapped bereits true ist.
func refreshDevices(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		// metal never updates free VRAM
		return append([]ml.DeviceInfo{}, devices...)
	}

	slog.Debug("refreshing free memory")
	updated := make([]bool, len(devices))
	allDone := func() bool {
		allDone := true
		for _, done := range updated {
			if !done {
				allDone = false
				break
			}
		}
		return allDone
	}

	// First try to use existing runners to refresh VRAM since they're already
	// active on GPU(s)
	for _, runner := range runners {
		if runner == nil {
			continue
		}
		deviceIDs := runner.GetActiveDeviceIDs()
		if len(deviceIDs) == 0 {
			// Skip this runner since it doesn't have active GPU devices
			continue
		}

		// Check to see if this runner is active on any devices that need a refresh
		skip := true
	devCheck:
		for _, dev := range deviceIDs {
			for i := range devices {
				if dev == devices[i].DeviceID {
					if !updated[i] {
						skip = false
						break devCheck
					}
				}
			}
		}
		if skip {
			continue
		}

		// Typical refresh on existing runner is ~500ms but allow longer if the system
		// is under stress before giving up and using stale data.
		ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		start := time.Now()
		updatedDevices := runner.GetDeviceInfos(ctx)
		slog.Debug("existing runner discovery took", "duration", time.Since(start))
		for _, u := range updatedDevices {
			for i := range devices {
				if u.DeviceID == devices[i].DeviceID {
					updated[i] = true
					devices[i].FreeMemory = u.FreeMemory
					break
				}
			}
		}
		// Short circuit if we've updated all the devices
		if allDone() {
			break
		}
	}
	if !allDone() {
		slog.Debug("unable to refresh all GPUs with existing runners, performing bootstrap discovery")

		// Bootstrapping may take longer in some cases (AMD windows), but we
		// would rather use stale free data to get the model running sooner
		ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()

		// Apply any dev filters to avoid re-discovering unsupported devices, and get IDs correct
		// We avoid CUDA filters here to keep ROCm from failing to discover GPUs in a mixed environment
		devFilter := ml.GetVisibleDevicesEnv(devices, false)

		for dir := range libDirs {
			updatedDevices := bootstrapDevices(ctx, []string{ml.LibOllamaPath, dir}, devFilter)
			for _, u := range updatedDevices {
				for i := range devices {
					if u.DeviceID == devices[i].DeviceID && u.PCIID == devices[i].PCIID {
						updated[i] = true
						devices[i].FreeMemory = u.FreeMemory
						break
					}
				}
				// TODO - consider evaluating if new devices have appeared (e.g. hotplug)
			}
			if allDone() {
				break
			}
		}
		if !allDone() {
			slog.Warn("unable to refresh free memory, using old values")
		}
	}

	return append([]ml.DeviceInfo{}, devices...)
}
