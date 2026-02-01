// Package siglip - Enum-Definitionen fuer SigLIP.
//
// Diese Datei enthaelt:
// - ModelType: Modell-Typ-Definitionen (ViT-B, ViT-L, ViT-SO400M)
// - Backend: Compute-Backend-Definitionen (CPU, CUDA, Metal, Vulkan)
// - LogLevel: Log-Level-Definitionen
// - EmbedFormat: Embedding-Format-Definitionen
// - Fehler-Definitionen
package siglip

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/src
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build -lsiglip -lggml -lm -lstdc++
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lpthread

#include "siglip.h"
*/
import "C"

import "errors"

// ============================================================================
// Fehler-Definitionen
// ============================================================================

var (
	ErrModelNotLoaded    = errors.New("siglip: model not loaded")
	ErrInvalidImage      = errors.New("siglip: invalid image data")
	ErrEncodingFailed    = errors.New("siglip: encoding failed")
	ErrInvalidParameters = errors.New("siglip: invalid parameters")
	ErrDimensionMismatch = errors.New("siglip: embedding dimension mismatch")
)

// ============================================================================
// ModelType Enum
// ============================================================================

// ModelType definiert den Modell-Typ.
type ModelType int

const (
	ModelVitB16    ModelType = C.SIGLIP_MODEL_VIT_B_16   // ViT-Base, Patch 16, 86M params
	ModelVitL16    ModelType = C.SIGLIP_MODEL_VIT_L_16   // ViT-Large, Patch 16, 303M params
	ModelVitSO400M ModelType = C.SIGLIP_MODEL_VIT_SO400M // ViT-SO400M, Patch 14, 400M params
	ModelUnknown   ModelType = C.SIGLIP_MODEL_UNKNOWN
)

// String gibt den Namen des Modell-Typs zurueck.
func (t ModelType) String() string {
	switch t {
	case ModelVitB16:
		return "ViT-B/16"
	case ModelVitL16:
		return "ViT-L/16"
	case ModelVitSO400M:
		return "ViT-SO400M"
	default:
		return "Unknown"
	}
}

// ============================================================================
// Backend Enum
// ============================================================================

// Backend definiert das Compute-Backend.
type Backend int

const (
	BackendCPU    Backend = C.SIGLIP_BACKEND_CPU    // CPU (GGML)
	BackendCUDA   Backend = C.SIGLIP_BACKEND_CUDA   // NVIDIA CUDA
	BackendMetal  Backend = C.SIGLIP_BACKEND_METAL  // Apple Metal
	BackendVulkan Backend = C.SIGLIP_BACKEND_VULKAN // Vulkan (experimentell)
)

// String gibt den Namen des Backends zurueck.
func (b Backend) String() string {
	switch b {
	case BackendCPU:
		return "CPU"
	case BackendCUDA:
		return "CUDA"
	case BackendMetal:
		return "Metal"
	case BackendVulkan:
		return "Vulkan"
	default:
		return "Unknown"
	}
}

// ============================================================================
// LogLevel Enum
// ============================================================================

// LogLevel definiert das Log-Level.
type LogLevel int

const (
	LogNone  LogLevel = C.SIGLIP_LOG_NONE
	LogError LogLevel = C.SIGLIP_LOG_ERROR
	LogWarn  LogLevel = C.SIGLIP_LOG_WARN
	LogInfo  LogLevel = C.SIGLIP_LOG_INFO
	LogDebug LogLevel = C.SIGLIP_LOG_DEBUG
)

// ============================================================================
// EmbedFormat Enum
// ============================================================================

// EmbedFormat definiert das Embedding-Format.
type EmbedFormat int

const (
	EmbedF32        EmbedFormat = C.SIGLIP_EMBED_F32        // float32 Array
	EmbedF16        EmbedFormat = C.SIGLIP_EMBED_F16        // float16 Array
	EmbedNormalized EmbedFormat = C.SIGLIP_EMBED_NORMALIZED // L2-normalisiert
)
