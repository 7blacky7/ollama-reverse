// Modul: runner_helpers.go
// Beschreibung: Hilfsfunktionen fuer GPU-Discovery.
// Enthaelt Bootstrap-Runner, Filter-Logik und Warnungen.

package discover

import (
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

func filterOverlapByLibrary(supported map[string]map[string]map[string]int, needsDelete []bool) {
	// For multi-GPU systems, use the newest version that supports all the GPUs
	for _, byLibDirs := range supported {
		libDirs := make([]string, 0, len(byLibDirs))
		for libDir := range byLibDirs {
			libDirs = append(libDirs, libDir)
		}
		sort.Sort(sort.Reverse(sort.StringSlice(libDirs)))
		anyMissing := false
		var newest string
		for _, newest = range libDirs {
			for _, libDir := range libDirs {
				if libDir == newest {
					continue
				}
				if len(byLibDirs[newest]) != len(byLibDirs[libDir]) {
					anyMissing = true
					break
				}
				for dev := range byLibDirs[newest] {
					if _, found := byLibDirs[libDir][dev]; !found {
						anyMissing = true
						break
					}
				}
			}
			if !anyMissing {
				break
			}
		}
		// Now we can mark overlaps for deletion
		for _, libDir := range libDirs {
			if libDir == newest {
				continue
			}
			for dev, i := range byLibDirs[libDir] {
				if _, found := byLibDirs[newest][dev]; found {
					slog.Debug("filtering device with overlapping libraries",
						"id", dev,
						"library", libDir,
						"delete_index", i,
						"kept_library", newest,
					)
					needsDelete[i] = true
				}
			}
		}
	}
}

type bootstrapRunner struct {
	port int
	cmd  *exec.Cmd
}

func (r *bootstrapRunner) GetPort() int {
	return r.port
}

func (r *bootstrapRunner) HasExited() bool {
	if r.cmd != nil && r.cmd.ProcessState != nil {
		return true
	}
	return false
}

func bootstrapDevices(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	var out io.Writer
	if envconfig.LogLevel() == logutil.LevelTrace {
		out = os.Stderr
	}
	start := time.Now()
	defer func() {
		slog.Debug("bootstrap discovery took", "duration", time.Since(start), "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs)
	}()

	logutil.Trace("starting runner for device discovery", "libDirs", ollamaLibDirs, "extraEnvs", extraEnvs)
	cmd, port, err := llm.StartRunner(
		true, // ollama engine
		"",   // no model
		ollamaLibDirs,
		out,
		extraEnvs,
	)
	if err != nil {
		slog.Debug("failed to start runner to discovery GPUs", "error", err)
		return nil
	}

	go func() {
		cmd.Wait() // exit status ignored
	}()

	defer cmd.Process.Kill()
	devices, err := ml.GetDevicesFromRunner(ctx, &bootstrapRunner{port: port, cmd: cmd})
	if err != nil {
		if cmd.ProcessState != nil && cmd.ProcessState.ExitCode() >= 0 {
			// Expected during bootstrapping while we filter out unsupported AMD GPUs
			logutil.Trace("runner exited", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs, "code", cmd.ProcessState.ExitCode())
		} else {
			slog.Info("failure during GPU discovery", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs, "error", err)
		}
	}
	logutil.Trace("runner enumerated devices", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "devices", devices)

	return devices
}

func overrideWarnings() {
	anyFound := false
	m := envconfig.AsMap()
	for _, k := range []string{
		"CUDA_VISIBLE_DEVICES",
		"HIP_VISIBLE_DEVICES",
		"ROCR_VISIBLE_DEVICES",
		"GGML_VK_VISIBLE_DEVICES",
		"GPU_DEVICE_ORDINAL",
		"HSA_OVERRIDE_GFX_VERSION",
	} {
		if e, found := m[k]; found && e.Value != "" {
			anyFound = true
			slog.Warn("user overrode visible devices", k, e.Value)
		}
	}
	if anyFound {
		slog.Warn("if GPUs are not correctly discovered, unset and try again")
	}
}

func detectIncompatibleLibraries() {
	if runtime.GOOS != "windows" {
		return
	}
	basePath, err := exec.LookPath("ggml-base.dll")
	if err != nil || basePath == "" {
		return
	}
	if !strings.HasPrefix(basePath, ml.LibOllamaPath) {
		slog.Warn("potentially incompatible library detected in PATH", "location", basePath)
	}
}
