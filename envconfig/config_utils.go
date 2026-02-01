// config_utils.go - Utility-Funktionen und Export fuer Konfiguration
//
// Dieses Modul enthaelt:
// - BoolWithDefault/Bool: Boolean-Getter mit Default-Wert
// - String: String-Getter
// - Uint/Uint64: Integer-Getter mit Default-Wert
// - EnvVar: Struktur fuer Environment-Variablen-Info
// - AsMap: Gibt alle Konfigurationen als Map zurueck
// - Values: Gibt alle Konfigurationswerte als String-Map zurueck
package envconfig

import (
	"fmt"
	"log/slog"
	"runtime"
	"strconv"
)

// =============================================================================
// Boolean-Getter
// =============================================================================

// BoolWithDefault gibt eine Funktion zurueck, die einen Bool mit Default-Wert liest
func BoolWithDefault(k string) func(defaultValue bool) bool {
	return func(defaultValue bool) bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}
			return b
		}
		return defaultValue
	}
}

// Bool gibt eine Funktion zurueck, die einen Bool liest (Default: false)
func Bool(k string) func() bool {
	withDefault := BoolWithDefault(k)
	return func() bool {
		return withDefault(false)
	}
}

// =============================================================================
// String-Getter
// =============================================================================

// String gibt eine Funktion zurueck, die einen String liest
func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

// =============================================================================
// Integer-Getter
// =============================================================================

// Uint gibt eine Funktion zurueck, die einen uint mit Default-Wert liest
func Uint(key string, defaultValue uint) func() uint {
	return func() uint {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return uint(n)
			}
		}
		return defaultValue
	}
}

// Uint64 gibt eine Funktion zurueck, die einen uint64 mit Default-Wert liest
func Uint64(key string, defaultValue uint64) func() uint64 {
	return func() uint64 {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return n
			}
		}
		return defaultValue
	}
}

// =============================================================================
// Export-Strukturen und -Funktionen
// =============================================================================

// EnvVar repraesentiert eine Environment-Variable mit Metadaten
type EnvVar struct {
	Name        string
	Value       any
	Description string
}

// AsMap gibt alle Konfigurationen als Map zurueck
// Enthaelt Namen, aktuelle Werte und Beschreibungen
func AsMap() map[string]EnvVar {
	ret := map[string]EnvVar{
		"OLLAMA_DEBUG":             {"OLLAMA_DEBUG", LogLevel(), "Show additional debug information (e.g. OLLAMA_DEBUG=1)"},
		"OLLAMA_FLASH_ATTENTION":   {"OLLAMA_FLASH_ATTENTION", FlashAttention(false), "Enabled flash attention"},
		"OLLAMA_KV_CACHE_TYPE":     {"OLLAMA_KV_CACHE_TYPE", KvCacheType(), "Quantization type for the K/V cache (default: f16)"},
		"OLLAMA_GPU_OVERHEAD":      {"OLLAMA_GPU_OVERHEAD", GpuOverhead(), "Reserve a portion of VRAM per GPU (bytes)"},
		"OLLAMA_HOST":              {"OLLAMA_HOST", Host(), "IP Address for the ollama server (default 127.0.0.1:11434)"},
		"OLLAMA_KEEP_ALIVE":        {"OLLAMA_KEEP_ALIVE", KeepAlive(), "The duration that models stay loaded in memory (default \"5m\")"},
		"OLLAMA_LLM_LIBRARY":       {"OLLAMA_LLM_LIBRARY", LLMLibrary(), "Set LLM library to bypass autodetection"},
		"OLLAMA_LOAD_TIMEOUT":      {"OLLAMA_LOAD_TIMEOUT", LoadTimeout(), "How long to allow model loads to stall before giving up (default \"5m\")"},
		"OLLAMA_MAX_LOADED_MODELS": {"OLLAMA_MAX_LOADED_MODELS", MaxRunners(), "Maximum number of loaded models per GPU"},
		"OLLAMA_MAX_QUEUE":         {"OLLAMA_MAX_QUEUE", MaxQueue(), "Maximum number of queued requests"},
		"OLLAMA_MODELS":            {"OLLAMA_MODELS", Models(), "The path to the models directory"},
		"OLLAMA_NOHISTORY":         {"OLLAMA_NOHISTORY", NoHistory(), "Do not preserve readline history"},
		"OLLAMA_NOPRUNE":           {"OLLAMA_NOPRUNE", NoPrune(), "Do not prune model blobs on startup"},
		"OLLAMA_NUM_PARALLEL":      {"OLLAMA_NUM_PARALLEL", NumParallel(), "Maximum number of parallel requests"},
		"OLLAMA_ORIGINS":           {"OLLAMA_ORIGINS", AllowedOrigins(), "A comma separated list of allowed origins"},
		"OLLAMA_SCHED_SPREAD":      {"OLLAMA_SCHED_SPREAD", SchedSpread(), "Always schedule model across all GPUs"},
		"OLLAMA_MULTIUSER_CACHE":   {"OLLAMA_MULTIUSER_CACHE", MultiUserCache(), "Optimize prompt caching for multi-user scenarios"},
		"OLLAMA_CONTEXT_LENGTH":    {"OLLAMA_CONTEXT_LENGTH", ContextLength(), "Context length to use unless otherwise specified (default: 4096)"},
		"OLLAMA_NEW_ENGINE":        {"OLLAMA_NEW_ENGINE", NewEngine(), "Enable the new Ollama engine"},
		"OLLAMA_REMOTES":           {"OLLAMA_REMOTES", Remotes(), "Allowed hosts for remote models (default \"ollama.com\")"},

		// Proxy-Einstellungen
		"HTTP_PROXY":  {"HTTP_PROXY", String("HTTP_PROXY")(), "HTTP proxy"},
		"HTTPS_PROXY": {"HTTPS_PROXY", String("HTTPS_PROXY")(), "HTTPS proxy"},
		"NO_PROXY":    {"NO_PROXY", String("NO_PROXY")(), "No proxy"},
	}

	// Nicht-Windows: Case-sensitive Proxy-Variablen
	if runtime.GOOS != "windows" {
		ret["http_proxy"] = EnvVar{"http_proxy", String("http_proxy")(), "HTTP proxy"}
		ret["https_proxy"] = EnvVar{"https_proxy", String("https_proxy")(), "HTTPS proxy"}
		ret["no_proxy"] = EnvVar{"no_proxy", String("no_proxy")(), "No proxy"}
	}

	// Nicht-macOS: GPU-Variablen
	if runtime.GOOS != "darwin" {
		ret["CUDA_VISIBLE_DEVICES"] = EnvVar{"CUDA_VISIBLE_DEVICES", CudaVisibleDevices(), "Set which NVIDIA devices are visible"}
		ret["HIP_VISIBLE_DEVICES"] = EnvVar{"HIP_VISIBLE_DEVICES", HipVisibleDevices(), "Set which AMD devices are visible by numeric ID"}
		ret["ROCR_VISIBLE_DEVICES"] = EnvVar{"ROCR_VISIBLE_DEVICES", RocrVisibleDevices(), "Set which AMD devices are visible by UUID or numeric ID"}
		ret["GGML_VK_VISIBLE_DEVICES"] = EnvVar{"GGML_VK_VISIBLE_DEVICES", VkVisibleDevices(), "Set which Vulkan devices are visible by numeric ID"}
		ret["GPU_DEVICE_ORDINAL"] = EnvVar{"GPU_DEVICE_ORDINAL", GpuDeviceOrdinal(), "Set which AMD devices are visible by numeric ID"}
		ret["HSA_OVERRIDE_GFX_VERSION"] = EnvVar{"HSA_OVERRIDE_GFX_VERSION", HsaOverrideGfxVersion(), "Override the gfx used for all detected AMD GPUs"}
		ret["OLLAMA_VULKAN"] = EnvVar{"OLLAMA_VULKAN", EnableVulkan(), "Enable experimental Vulkan support"}
	}

	return ret
}

// Values gibt alle Konfigurationswerte als String-Map zurueck
func Values() map[string]string {
	vals := make(map[string]string)
	for k, v := range AsMap() {
		vals[k] = fmt.Sprintf("%v", v.Value)
	}
	return vals
}
