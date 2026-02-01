// tensortype.go - GGML TensorType Definitionen
// Enthält: TensorType Konstanten, Parsing und Hilfsfunktionen

package ggml

import (
	"fmt"
)

// TensorType ist äquivalent zu ggml_type für einzelne Tensor-Typen
// Hinweis: Diese sind nicht identisch mit FileType
type TensorType uint32

const (
	TensorTypeF32 TensorType = iota
	TensorTypeF16
	TensorTypeQ4_0
	TensorTypeQ4_1
	tensorTypeQ4_2
	tensorTypeQ4_3 // unbenutzt
	TensorTypeQ5_0
	TensorTypeQ5_1
	TensorTypeQ8_0
	TensorTypeQ8_1
	TensorTypeQ2_K
	TensorTypeQ3_K
	TensorTypeQ4_K
	TensorTypeQ5_K
	TensorTypeQ6_K
	TensorTypeQ8_K
	tensorTypeIQ2_XXS // nicht unterstützt
	tensorTypeIQ2_XS  // nicht unterstützt
	tensorTypeIQ3_XXS // nicht unterstützt
	tensorTypeIQ1_S   // nicht unterstützt
	tensorTypeIQ4_NL  // nicht unterstützt
	tensorTypeIQ3_S   // nicht unterstützt
	tensorTypeIQ2_S   // nicht unterstützt
	tensorTypeIQ4_XS  // nicht unterstützt
	TensorTypeI8
	TensorTypeI16
	TensorTypeI32
	TensorTypeI64
	TensorTypeF64
	tensorTypeIQ1_M // nicht unterstützt
	TensorTypeBF16
	tensorTypeQ4_0_4_4   // unbenutzt
	tensorTypeQ4_0_4_8   // unbenutzt
	tensorTypeQ4_0_8_8   // unbenutzt
	tensorTypeTQ1_0      // nicht unterstützt
	tensorTypeTQ2_0      // nicht unterstützt
	tensorTypeIQ4_NL_4_4 // unbenutzt
	tensorTypeIQ4_NL_4_8 // unbenutzt
	tensorTypeIQ4_NL_8_8 // unbenutzt
	TensorTypeMXFP4
)

// ParseTensorType parst den GGUF-Tensortyp aus einem String
// Nur von Ollama unterstützte Typen werden als gültig betrachtet
func ParseTensorType(s string) (TensorType, error) {
	switch s {
	case "F32":
		return TensorTypeF32, nil
	case "F16":
		return TensorTypeF16, nil
	case "Q4_0":
		return TensorTypeQ4_0, nil
	case "Q4_1":
		return TensorTypeQ4_1, nil
	case "Q5_0":
		return TensorTypeQ5_0, nil
	case "Q5_1":
		return TensorTypeQ5_1, nil
	case "Q8_0":
		return TensorTypeQ8_0, nil
	case "Q8_1":
		return TensorTypeQ8_1, nil
	case "Q2_K":
		return TensorTypeQ2_K, nil
	case "Q3_K":
		return TensorTypeQ3_K, nil
	case "Q4_K":
		return TensorTypeQ4_K, nil
	case "Q5_K":
		return TensorTypeQ5_K, nil
	case "Q6_K":
		return TensorTypeQ6_K, nil
	case "Q8_K":
		return TensorTypeQ8_K, nil
	case "F64":
		return TensorTypeF64, nil
	case "BF16":
		return TensorTypeBF16, nil
	case "MXFP4":
		return TensorTypeMXFP4, nil
	default:
		return 0, fmt.Errorf("unsupported quantization type %s", s)
	}
}

// IsQuantized prüft ob der TensorType quantisiert ist
func (t TensorType) IsQuantized() bool {
	switch t {
	case TensorTypeF32, TensorTypeF16, TensorTypeBF16:
		return false
	default:
		return true
	}
}

// RowSize berechnet die Zeilengröße in Bytes
func (t TensorType) RowSize(ne uint64) uint64 {
	return t.TypeSize() * ne / t.BlockSize()
}

// String gibt die String-Repräsentation des TensorType zurück
func (t TensorType) String() string {
	switch t {
	case TensorTypeF32:
		return "F32"
	case TensorTypeF16:
		return "F16"
	case TensorTypeQ4_0:
		return "Q4_0"
	case TensorTypeQ4_1:
		return "Q4_1"
	case TensorTypeQ5_0:
		return "Q5_0"
	case TensorTypeQ5_1:
		return "Q5_1"
	case TensorTypeQ8_0:
		return "Q8_0"
	case TensorTypeQ8_1:
		return "Q8_1"
	case TensorTypeQ2_K:
		return "Q2_K"
	case TensorTypeQ3_K:
		return "Q3_K"
	case TensorTypeQ4_K:
		return "Q4_K"
	case TensorTypeQ5_K:
		return "Q5_K"
	case TensorTypeQ6_K:
		return "Q6_K"
	case TensorTypeQ8_K:
		return "Q8_K"
	case TensorTypeF64:
		return "F64"
	case TensorTypeBF16:
		return "BF16"
	case 4, TensorTypeMXFP4:
		return "MXFP4"
	default:
		return "unknown"
	}
}
