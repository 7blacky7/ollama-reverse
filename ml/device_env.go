// device_env.go
// Dieses Modul enthaelt Funktionen fuer Umgebungsvariablen und
// Flash-Attention-Unterstuetzung bei GPU-Geraeten.
// Urspruenglich aus device.go extrahiert.

package ml

import (
	"log/slog"
	"runtime"
)

// For each GPU, check if it does NOT support flash attention
func FlashAttentionSupported(l []DeviceInfo) bool {
	for _, gpu := range l {
		supportsFA := gpu.Library == "cpu" ||
			gpu.Name == "Metal" || gpu.Library == "Metal" ||
			(gpu.Library == "CUDA" && gpu.DriverMajor >= 7 && !(gpu.ComputeMajor == 7 && gpu.ComputeMinor == 2)) ||
			gpu.Library == "ROCm" ||
			gpu.Library == "Vulkan"

		if !supportsFA {
			return false
		}
	}
	return true
}

type FlashAttentionType int32

const (
	// Aligned with llama_flash_attn_type
	FlashAttentionAuto     FlashAttentionType = -1
	FlashAttentionDisabled FlashAttentionType = 0
	FlashAttentionEnabled  FlashAttentionType = 1
)

func (f FlashAttentionType) LogValue() slog.Value {
	return slog.AnyValue(f.String())
}

func (f FlashAttentionType) String() string {
	switch f {
	case FlashAttentionAuto:
		return "Auto"
	case FlashAttentionDisabled:
		return "Disabled"
	case FlashAttentionEnabled:
		return "Enabled"
	default:
		return "unknown"
	}
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variables
// Set mustFilter true to enable filtering of CUDA devices
func GetVisibleDevicesEnv(l []DeviceInfo, mustFilter bool) map[string]string {
	if len(l) == 0 {
		return nil
	}
	env := map[string]string{}
	for _, d := range l {
		d.updateVisibleDevicesEnv(env, mustFilter)
	}
	return env
}

// NeedsInitValidation returns true if the device in question has the potential
// to crash at inference time and requires deeper validation before we include
// it in the supported devices list.
func (d DeviceInfo) NeedsInitValidation() bool {
	// ROCm: rocblas will crash on unsupported devices.
	// CUDA: verify CC is supported by the version of the library
	return d.Library == "ROCm" || d.Library == "CUDA"
}

// Set the init validation environment variable
func (d DeviceInfo) AddInitValidation(env map[string]string) {
	env["GGML_CUDA_INIT"] = "1" // force deep initialization to trigger crash on unsupported GPUs
}

// PreferredLibrary returns true if this library is preferred over the other input
// library
// Used to filter out Vulkan in favor of CUDA or ROCm
func (d DeviceInfo) PreferredLibrary(other DeviceInfo) bool {
	// TODO in the future if we find Vulkan is better than ROCm on some devices
	// that implementation can live here.

	if d.Library == "CUDA" || d.Library == "ROCm" {
		return true
	}
	return false
}

func (d DeviceInfo) updateVisibleDevicesEnv(env map[string]string, mustFilter bool) {
	var envVar string
	switch d.Library {
	case "ROCm":
		// ROCm must be filtered as it can crash the runner on unsupported devices
		envVar = "ROCR_VISIBLE_DEVICES"
		if runtime.GOOS != "linux" {
			envVar = "HIP_VISIBLE_DEVICES"
		}
	case "CUDA":
		if !mustFilter {
			// By default we try to avoid filtering CUDA devices because ROCm also
			// looks at the CUDA env var, and gets confused in mixed vendor environments.
			return
		}
		envVar = "CUDA_VISIBLE_DEVICES"
	default:
		// Vulkan is not filtered via env var, but via scheduling decisions
		return
	}
	v, existing := env[envVar]
	if existing {
		v = v + ","
	}
	if d.FilterID != "" {
		v = v + d.FilterID
	} else {
		v = v + d.ID
	}
	env[envVar] = v
}
