// config_features.go - Feature-Flags und GPU-Konfiguration
//
// Dieses Modul enthaelt:
// - Feature-Flags (FlashAttention, NewEngine, etc.)
// - GPU-bezogene Environment-Variablen
// - Parallelitaets-Einstellungen
package envconfig

// =============================================================================
// Feature-Flags
// =============================================================================

var (
	// FlashAttention aktiviert experimentelles Flash Attention
	FlashAttention = BoolWithDefault("OLLAMA_FLASH_ATTENTION")

	// KvCacheType ist der Quantisierungstyp fuer den K/V Cache
	KvCacheType = String("OLLAMA_KV_CACHE_TYPE")

	// NoHistory deaktiviert Readline-History
	NoHistory = Bool("OLLAMA_NOHISTORY")

	// NoPrune deaktiviert Pruning von Model-Blobs beim Start
	NoPrune = Bool("OLLAMA_NOPRUNE")

	// SchedSpread erlaubt Scheduling von Models ueber alle GPUs
	SchedSpread = Bool("OLLAMA_SCHED_SPREAD")

	// MultiUserCache optimiert Prompt-Caching fuer Multi-User-Szenarien
	MultiUserCache = Bool("OLLAMA_MULTIUSER_CACHE")

	// NewEngine aktiviert die neue Ollama-Engine
	NewEngine = Bool("OLLAMA_NEW_ENGINE")

	// ContextLength setzt die Standard-Context-Laenge
	ContextLength = Uint("OLLAMA_CONTEXT_LENGTH", 4096)

	// UseAuth aktiviert Authentifizierung zwischen Client und Server
	UseAuth = Bool("OLLAMA_AUTH")

	// EnableVulkan aktiviert experimentelles Vulkan-Backend
	EnableVulkan = Bool("OLLAMA_VULKAN")
)

// =============================================================================
// LLM-Library Konfiguration
// =============================================================================

var (
	// LLMLibrary ueberschreibt die automatische Library-Erkennung
	LLMLibrary = String("OLLAMA_LLM_LIBRARY")
)

// =============================================================================
// GPU-Sichtbarkeits-Variablen
// =============================================================================

var (
	// CudaVisibleDevices steuert sichtbare NVIDIA-Geraete
	CudaVisibleDevices = String("CUDA_VISIBLE_DEVICES")

	// HipVisibleDevices steuert sichtbare AMD-Geraete (numerische ID)
	HipVisibleDevices = String("HIP_VISIBLE_DEVICES")

	// RocrVisibleDevices steuert sichtbare AMD-Geraete (UUID oder numerische ID)
	RocrVisibleDevices = String("ROCR_VISIBLE_DEVICES")

	// VkVisibleDevices steuert sichtbare Vulkan-Geraete (numerische ID)
	VkVisibleDevices = String("GGML_VK_VISIBLE_DEVICES")

	// GpuDeviceOrdinal steuert sichtbare AMD-Geraete (numerische ID)
	GpuDeviceOrdinal = String("GPU_DEVICE_ORDINAL")

	// HsaOverrideGfxVersion ueberschreibt die GFX-Version fuer AMD-GPUs
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)

// =============================================================================
// Parallelitaets- und Queue-Einstellungen
// =============================================================================

var (
	// NumParallel setzt die Anzahl paralleler Model-Requests
	// Konfigurierbar via OLLAMA_NUM_PARALLEL
	NumParallel = Uint("OLLAMA_NUM_PARALLEL", 1)

	// MaxRunners setzt die maximale Anzahl geladener Models
	// Konfigurierbar via OLLAMA_MAX_LOADED_MODELS
	MaxRunners = Uint("OLLAMA_MAX_LOADED_MODELS", 0)

	// MaxQueue setzt die maximale Anzahl wartender Requests
	// Konfigurierbar via OLLAMA_MAX_QUEUE
	MaxQueue = Uint("OLLAMA_MAX_QUEUE", 512)
)

// =============================================================================
// GPU-Speicher-Einstellungen
// =============================================================================

var (
	// GpuOverhead reserviert VRAM pro GPU (in Bytes)
	// Konfigurierbar via OLLAMA_GPU_OVERHEAD
	GpuOverhead = Uint64("OLLAMA_GPU_OVERHEAD", 0)
)
