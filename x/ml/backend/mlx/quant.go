//go:build mlx

// quant.go - Quantisierungs-Funktionen fuer MLX-Backend
//
// Dieses Modul enthaelt:
// - gguf_load_quantized: Laedt Q4_0, Q4_1, Q8_0 quantisierte Tensoren
// - load_k_quantized: Laedt K-quantisierte Tensoren (Q4_K, Q6_K)
//
// Die C-Code-Definitionen fuer die Extraktions-Funktionen sind
// in quant_c.go ausgelagert.
package mlx

/*
#include "quant_c.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/x448/float16"
)

// gguf_load_quantized laedt quantisierte GGUF-Tensoren
// Unterstuetzte Typen:
// - dtype 2: Q4_0 (4-bit mit Scale)
// - dtype 3: Q4_1 (4-bit mit Scale und Bias)
// - dtype 8: Q8_0 (8-bit mit Scale)
func gguf_load_quantized(data unsafe.Pointer, name string, final_shape []C.int, dtype uint32, stream C.mlx_stream) (r C.mlx_array, err error) {
	shape := append([]C.int{}, final_shape...)

	// Weights pro Byte bestimmen
	var weights_per_byte C.int
	if dtype == 2 || dtype == 3 {
		weights_per_byte = 2
	} else if dtype == 8 {
		weights_per_byte = 1
	} else {
		return r, fmt.Errorf("unsupported tensor type %d", dtype)
	}

	// Shape validieren
	weights_per_block := C.int(32)
	if shape[len(shape)-1]%weights_per_block != 0 {
		return r, fmt.Errorf("[load_gguf] tensor has incompatible last dim shape: %d", shape[len(shape)-1])
	}

	// Weights-Array erstellen
	weights_shape := append([]C.int{}, shape...)
	weights_shape[len(weights_shape)-1] /= (weights_per_byte * 4)
	w_nbytes := C.int(unsafe.Sizeof(uint32(0)))
	for i := range weights_shape {
		w_nbytes *= weights_shape[i]
	}
	w_data := make([]byte, w_nbytes)
	cbytes := C.CBytes(w_data)
	defer C.free(cbytes)
	weights := C.mlx_array_new_data(
		cbytes,
		&weights_shape[0],
		C.int(len(weights_shape)),
		C.MLX_UINT32,
	)

	// Scales und Bias Arrays erstellen
	shape[len(shape)-1] = shape[len(shape)-1] / weights_per_block
	sb_nbytes := C.int(unsafe.Sizeof(float16.Float16(0)))
	for i := range shape {
		sb_nbytes *= shape[i]
	}

	// Scales Array
	s_data := make([]byte, sb_nbytes)
	cbytes = C.CBytes(s_data)
	defer C.free(cbytes)
	scales := C.mlx_array_new_data(
		cbytes,
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)

	// Bias Array
	b_data := make([]byte, sb_nbytes)
	cbytes = C.CBytes(b_data)
	defer C.free(cbytes)
	biases := C.mlx_array_new_data(
		cbytes,
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)

	// Daten extrahieren je nach Typ
	var bits C.int
	switch dtype {
	case 2:
		C.extract_q4_0_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 4
	case 3:
		C.extract_q4_1_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 4
	case 8:
		C.extract_q8_0_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 8
	}

	// Dequantisieren
	groupSize := C.mlx_optional_int{value: 32, has_value: true}
	bitsOpt := C.mlx_optional_int{value: bits, has_value: true}
	var dtypeOpt C.mlx_optional_dtype // has_value defaults to false
	C.mlx_dequantize(
		&r,
		weights,
		scales,
		biases,
		groupSize,
		bitsOpt,
		nil, // TODO mode
		dtypeOpt,
		stream,
	)

	// Aufräumen
	C.mlx_array_free(weights)
	C.mlx_array_free(scales)
	C.mlx_array_free(biases)

	return r, nil
}

// load_k_quantized laedt K-quantisierte Tensoren
// Unterstuetzte Typen:
// - dtype 12: Q4_K (4-bit K-Quantisierung)
// - dtype 14: Q6_K (6-bit K-Quantisierung)
func load_k_quantized(data unsafe.Pointer, name string, shape []C.int, dtype uint32, stream C.mlx_stream) (r C.mlx_array, err error) {
	// Gesamtgroesse berechnen
	size := 1
	for _, d := range shape {
		size *= int(d)
	}

	// Float16-Puffer fuer dequantisierte Daten
	fdata := make([]float16.Float16, size)

	// Dequantisierung je nach Typ
	switch dtype {
	case 14:
		C.dequant_row_q6_K(
			data,
			unsafe.Pointer(&fdata[0]),
			C.int(size),
		)
	case 12:
		C.dequant_row_q4_K(
			data,
			unsafe.Pointer(&fdata[0]),
			C.int(size),
		)
	default:
		return r, fmt.Errorf("unsupported K quant")
	}

	// MLX-Array erstellen
	r = C.mlx_array_new_data(
		unsafe.Pointer(&fdata[0]),
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)

	return r, nil
}
