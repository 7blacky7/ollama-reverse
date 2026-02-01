// filetype.go - GGUF FileType Definitionen
// Enthält: FileType Konstanten, Parsing und Konvertierung zu TensorType

package ggml

import (
	"fmt"
	"log/slog"
	"strings"
)

// FileType ist der Go-Äquivalent zu llama_ftype für GGUF-Dateitypen
type FileType uint32

const (
	FileTypeF32 FileType = iota
	FileTypeF16
	fileTypeQ4_0
	fileTypeQ4_1
	fileTypeMXFP4 // ursprünglich fileTypeQ4_1_F16 // unbenutzt
	fileTypeQ4_2  // unbenutzt
	fileTypeQ4_3  // unbenutzt
	FileTypeQ8_0
	fileTypeQ5_0
	fileTypeQ5_1
	fileTypeQ2_K
	fileTypeQ3_K_S
	fileTypeQ3_K_M
	fileTypeQ3_K_L
	FileTypeQ4_K_S
	FileTypeQ4_K_M
	fileTypeQ5_K_S
	fileTypeQ5_K_M
	fileTypeQ6_K
	fileTypeIQ2_XXS
	fileTypeIQ2_XS
	fileTypeQ2_K_S
	fileTypeIQ3_XS
	fileTypeIQ3_XXS
	fileTypeIQ1_S
	fileTypeIQ4_NL
	fileTypeIQ3_S
	fileTypeIQ3_M
	fileTypeIQ2_S
	fileTypeIQ2_M
	fileTypeIQ4_XS
	fileTypeIQ1_M
	FileTypeBF16
	fileTypeQ4_0_4_4 // unbenutzt
	fileTypeQ4_0_4_8 // unbenutzt
	fileTypeQ4_0_8_8 // unbenutzt
	fileTypeTQ1_0
	fileTypeTQ2_0

	FileTypeUnknown = 1024
)

// ParseFileType parst den GGUF-Dateityp aus einem String
// Nur von Ollama unterstützte Typen werden als gültig betrachtet
func ParseFileType(s string) (FileType, error) {
	switch s {
	case "F32":
		return FileTypeF32, nil
	case "F16":
		return FileTypeF16, nil
	case "Q8_0":
		return FileTypeQ8_0, nil
	case "Q4_K_S":
		return FileTypeQ4_K_S, nil
	case "Q4_K_M", "Q4_K":
		return FileTypeQ4_K_M, nil
	case "BF16":
		return FileTypeBF16, nil
	default:
		supportedFileTypes := []FileType{
			FileTypeF32,
			FileTypeF16,
			FileTypeQ4_K_S,
			FileTypeQ4_K_M,
			FileTypeQ8_0,
		}
		strs := make([]string, len(supportedFileTypes))
		for i := range supportedFileTypes {
			strs[i] = supportedFileTypes[i].String()
		}

		return FileTypeUnknown, fmt.Errorf("unsupported quantization type %s - supported types are %s", s, strings.Join(strs, ", "))
	}
}

// String gibt die String-Repräsentation des FileType zurück
func (t FileType) String() string {
	switch t {
	case FileTypeF32:
		return "F32"
	case FileTypeF16:
		return "F16"
	case fileTypeQ4_0:
		return "Q4_0"
	case fileTypeQ4_1:
		return "Q4_1"
	case fileTypeMXFP4:
		return "MXFP4"
	case FileTypeQ8_0:
		return "Q8_0"
	case fileTypeQ5_0:
		return "Q5_0"
	case fileTypeQ5_1:
		return "Q5_1"
	case fileTypeQ2_K:
		return "Q2_K"
	case fileTypeQ3_K_S:
		return "Q3_K_S"
	case fileTypeQ3_K_M:
		return "Q3_K_M"
	case fileTypeQ3_K_L:
		return "Q3_K_L"
	case FileTypeQ4_K_S:
		return "Q4_K_S"
	case FileTypeQ4_K_M:
		return "Q4_K_M"
	case fileTypeQ5_K_S:
		return "Q5_K_S"
	case fileTypeQ5_K_M:
		return "Q5_K_M"
	case fileTypeQ6_K:
		return "Q6_K"
	case fileTypeQ2_K_S:
		return "Q2_K_S"
	case FileTypeBF16:
		return "BF16"
	default:
		return "unknown"
	}
}

// Value gibt den uint32-Wert des FileType zurück
func (t FileType) Value() uint32 {
	return uint32(t)
}

// ToTensorType konvertiert FileType zu TensorType
func (ftype FileType) ToTensorType() TensorType {
	switch ftype {
	case FileTypeF32:
		return TensorTypeF32
	case FileTypeF16:
		return TensorTypeF16
	case fileTypeQ4_0:
		return TensorTypeQ4_0
	case fileTypeQ4_1:
		return TensorTypeQ4_1
	case FileTypeQ8_0:
		return TensorTypeQ8_0
	case fileTypeQ5_0:
		return TensorTypeQ5_0
	case fileTypeQ5_1:
		return TensorTypeQ5_1
	case fileTypeQ2_K:
		return TensorTypeQ2_K
	case fileTypeQ3_K_S:
		return TensorTypeQ3_K
	case fileTypeQ3_K_M:
		return TensorTypeQ3_K
	case fileTypeQ3_K_L:
		return TensorTypeQ3_K
	case FileTypeQ4_K_S:
		return TensorTypeQ4_K
	case FileTypeQ4_K_M:
		return TensorTypeQ4_K
	case fileTypeQ5_K_S:
		return TensorTypeQ5_K
	case fileTypeQ5_K_M:
		return TensorTypeQ5_K
	case fileTypeQ6_K:
		return TensorTypeQ6_K
	case fileTypeQ2_K_S:
		return TensorTypeQ2_K
	case FileTypeBF16:
		return TensorTypeBF16
	case fileTypeMXFP4:
		return TensorTypeMXFP4
	default:
		slog.Warn("unsupported file type", "type", ftype)
		return 0 // F32
	}
}
