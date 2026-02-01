// Modul: runner_discovery.go
// Beschreibung: Haupt-GPU-Discovery-Logik mit Bootstrap-Phase.
// Enthaelt GPUDevices-Funktion und initiale GPU-Erkennung.

package discover

// Runner based GPU discovery

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

var (
	deviceMu     sync.Mutex
	devices      []ml.DeviceInfo
	libDirs      map[string]struct{}
	exe          string
	bootstrapped bool
)

func GPUDevices(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
	deviceMu.Lock()
	defer deviceMu.Unlock()
	startDiscovery := time.Now()
	msg := "overall device VRAM discovery took"
	defer func() {
		slog.Debug(msg, "duration", time.Since(startDiscovery))
	}()

	if !bootstrapped {
		msg = "GPU bootstrap discovery took"
		libDirs = make(map[string]struct{})
		var err error
		exe, err = os.Executable()
		if err != nil {
			slog.Error("unable to lookup executable path", "error", err)
			return nil
		}
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
		if err != nil {
			slog.Debug("unable to lookup runner library directories", "error", err)
		}
		for _, file := range files {
			libDirs[filepath.Dir(file)] = struct{}{}
		}

		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		slog.Info("discovering available GPUs...")
		detectIncompatibleLibraries()

		// Warn if any user-overrides are set which could lead to incorrect GPU discovery
		overrideWarnings()

		requested := envconfig.LLMLibrary()
		jetpack := cudaJetpack()

		// For our initial discovery pass, we gather all the known GPUs through
		// all the libraries that were detected. This pass may include GPUs that
		// are enumerated, but not actually supported.
		// We run this in serial to avoid potentially initializing a GPU multiple
		// times concurrently leading to memory contention
		for dir := range libDirs {
			// Typically bootstrapping takes < 1s, but on some systems, with devices
			// in low power/idle mode, initialization can take multiple seconds.  We
			// set a longer timeout just for bootstrap discovery to reduce the chance
			// of giving up too quickly
			bootstrapTimeout := 30 * time.Second
			if runtime.GOOS == "windows" {
				// On Windows with Defender enabled, AV scanning of the DLLs
				// takes place sequentially and this can significantly increase
				// the time it takes too do the initial discovery pass.
				// Subsequent loads will be faster as the scan results are
				// cached
				bootstrapTimeout = 90 * time.Second
			}
			var dirs []string
			if dir != "" {
				if requested != "" && filepath.Base(dir) != requested {
					slog.Debug("skipping available library at user's request", "requested", requested, "libDir", dir)
					continue
				} else if jetpack != "" && filepath.Base(dir) != "cuda_"+jetpack {
					continue
				} else if jetpack == "" && strings.Contains(filepath.Base(dir), "cuda_jetpack") {
					slog.Debug("jetpack not detected (set JETSON_JETPACK or OLLAMA_LLM_LIBRARY to override), skipping", "libDir", dir)
					continue
				} else if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
					slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1")
					continue
				}
				dirs = []string{ml.LibOllamaPath, dir}
			} else {
				dirs = []string{ml.LibOllamaPath}
			}

			ctx1stPass, cancel := context.WithTimeout(ctx, bootstrapTimeout)
			defer cancel()

			// For this pass, we retain duplicates in case any are incompatible with some libraries
			devices = append(devices, bootstrapDevices(ctx1stPass, dirs, nil)...)
		}

		// In the second pass, we more deeply initialize the GPUs to weed out devices that
		// aren't supported by a given library.  We run this phase in parallel to speed up discovery.
		// Only devices that need verification are included in this pass
		slog.Debug("evaluating which, if any, devices to filter out", "initial_count", len(devices))
		ctx2ndPass, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		var wg sync.WaitGroup
		needsDelete := make([]bool, len(devices))
		supportedMu := sync.Mutex{}
		supported := make(map[string]map[string]map[string]int) // [Library][libDir][ID] = pre-deletion devices index
		for i := range devices {
			libDir := devices[i].LibraryPath[len(devices[i].LibraryPath)-1]
			if !devices[i].NeedsInitValidation() {
				// No need to validate, add to the supported map
				supportedMu.Lock()
				if _, ok := supported[devices[i].Library]; !ok {
					supported[devices[i].Library] = make(map[string]map[string]int)
				}
				if _, ok := supported[devices[i].Library][libDir]; !ok {
					supported[devices[i].Library][libDir] = make(map[string]int)
				}
				supported[devices[i].Library][libDir][devices[i].ID] = i
				supportedMu.Unlock()
				continue
			}
			slog.Debug("verifying if device is supported", "library", libDir, "description", devices[i].Description, "compute", devices[i].Compute(), "id", devices[i].ID, "pci_id", devices[i].PCIID)
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				extraEnvs := ml.GetVisibleDevicesEnv(devices[i:i+1], true)
				devices[i].AddInitValidation(extraEnvs)
				if len(bootstrapDevices(ctx2ndPass, devices[i].LibraryPath, extraEnvs)) == 0 {
					slog.Debug("filtering device which didn't fully initialize",
						"id", devices[i].ID,
						"libdir", devices[i].LibraryPath[len(devices[i].LibraryPath)-1],
						"pci_id", devices[i].PCIID,
						"library", devices[i].Library,
					)
					needsDelete[i] = true
				} else {
					supportedMu.Lock()
					if _, ok := supported[devices[i].Library]; !ok {
						supported[devices[i].Library] = make(map[string]map[string]int)
					}
					if _, ok := supported[devices[i].Library][libDir]; !ok {
						supported[devices[i].Library][libDir] = make(map[string]int)
					}
					supported[devices[i].Library][libDir][devices[i].ID] = i
					supportedMu.Unlock()
				}
			}(i)
		}
		wg.Wait()
		logutil.Trace("supported GPU library combinations before filtering", "supported", supported)

		// Mark for deletion any overlaps - favoring the library version that can cover all GPUs if possible
		filterOverlapByLibrary(supported, needsDelete)

		// Any Libraries that utilize numeric IDs need adjusting based on any possible filtering taking place
		postFilteredID := map[string]int{}
		for i := 0; i < len(needsDelete); i++ {
			if needsDelete[i] {
				logutil.Trace("removing unsupported or overlapping GPU combination", "libDir", devices[i].LibraryPath[len(devices[i].LibraryPath)-1], "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
				devices = append(devices[:i], devices[i+1:]...)
				needsDelete = append(needsDelete[:i], needsDelete[i+1:]...)
				i--
			} else {
				if _, ok := postFilteredID[devices[i].Library]; !ok {
					postFilteredID[devices[i].Library] = 0
				}
				if _, err := strconv.Atoi(devices[i].ID); err == nil {
					// Replace the numeric ID with the post-filtered IDs
					slog.Debug("adjusting filtering IDs", "FilterID", devices[i].ID, "new_ID", strconv.Itoa(postFilteredID[devices[i].Library]))
					devices[i].FilterID = devices[i].ID
					devices[i].ID = strconv.Itoa(postFilteredID[devices[i].Library])
				}
				postFilteredID[devices[i].Library]++
			}
		}

		// Now filter out any overlap with different libraries (favor CUDA/HIP over others)
		for i := 0; i < len(devices); i++ {
			for j := i + 1; j < len(devices); j++ {
				// For this pass, we only drop exact duplicates
				switch devices[i].Compare(devices[j]) {
				case ml.SameBackendDevice:
					// Same library and device, skip it
					devices = append(devices[:j], devices[j+1:]...)
					j--
					continue
				case ml.DuplicateDevice:
					// Different library, choose based on priority
					var droppedDevice ml.DeviceInfo
					if devices[i].PreferredLibrary(devices[j]) {
						droppedDevice = devices[j]
					} else {
						droppedDevice = devices[i]
						devices[i] = devices[j]
					}
					devices = append(devices[:j], devices[j+1:]...)
					j--

					typeStr := "discrete"
					if droppedDevice.Integrated {
						typeStr = "iGPU"
					}
					slog.Debug("dropping duplicate device",
						"id", droppedDevice.ID,
						"library", droppedDevice.Library,
						"compute", droppedDevice.Compute(),
						"name", droppedDevice.Name,
						"description", droppedDevice.Description,
						"libdirs", strings.Join(droppedDevice.LibraryPath, ","),
						"driver", droppedDevice.Driver(),
						"pci_id", droppedDevice.PCIID,
						"type", typeStr,
						"total", format.HumanBytes2(droppedDevice.TotalMemory),
						"available", format.HumanBytes2(droppedDevice.FreeMemory),
					)
					continue
				}
			}
		}

		// Reset the libDirs to what we actually wind up using for future refreshes
		libDirs = make(map[string]struct{})
		for _, dev := range devices {
			dir := dev.LibraryPath[len(dev.LibraryPath)-1]
			if dir != ml.LibOllamaPath {
				libDirs[dir] = struct{}{}
			}
		}
		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		bootstrapped = true
	} else {
		return refreshDevices(ctx, runners)
	}

	return append([]ml.DeviceInfo{}, devices...)
}
