// Package ggml - Tensor Datenstrukturen
//
// Dieses Modul enthaelt Tensor-bezogene Typen und Methoden:
// - Tensor: Einzelner Tensor mit Name, Shape, Kind
// - Tensors: Collection von Tensors mit Offset
// - Layer: Map von Tensor-Namen zu Tensors
// - TensorType Methoden: BlockSize, TypeSize
package ggml

import (
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
)

// Tensors repraesentiert eine Sammlung von Tensors
type Tensors struct {
	items  []*Tensor
	Offset uint64
}

// Items gibt Tensors zurueck, optional gefiltert nach Prefix
func (s Tensors) Items(prefix ...string) []*Tensor {
	if len(prefix) == 0 {
		return s.items
	}

	var items []*Tensor
	for _, t := range s.items {
		if strings.HasPrefix(t.Name, prefix[0]) {
			items = append(items, t)
		}
	}

	return items
}

// GroupLayers gruppiert Tensors nach Layer-Namen
func (ts Tensors) GroupLayers() map[string]Layer {
	layers := make(map[string]Layer)
	for _, t := range ts.items {
		parts := strings.Split(t.Name, ".")
		if index := slices.IndexFunc(parts, func(s string) bool { return s == "blk" || s == "mm" }); index != -1 {
			if len(parts) > index+2 {
				// blk und mm sollten eine Nummer haben, diese joinen
				parts = append(
					[]string{strings.Join(parts[:index+2], ".")},
					parts[index+2:]...)
			}
		}

		if _, ok := layers[parts[0]]; !ok {
			layers[parts[0]] = make(Layer)
		}

		layers[parts[0]][strings.Join(parts[1:], ".")] = t
	}

	return layers
}

// Layer repraesentiert eine Gruppe von Tensors (z.B. ein Transformer-Block)
type Layer map[string]*Tensor

// Size berechnet die Gesamtgroesse aller Tensors im Layer
func (l Layer) Size() (size uint64) {
	for _, t := range l {
		size += t.Size()
	}
	return size
}

// Tensor repraesentiert einen einzelnen GGML-Tensor
type Tensor struct {
	Name   string `json:"name"`
	Kind   uint32 `json:"kind"`
	Offset uint64 `json:"-"`

	// Shape ist die Anzahl der Elemente in jeder Dimension
	Shape []uint64 `json:"shape"`

	io.WriterTo `json:"-"`
}

// block extrahiert die Block-Nummer aus dem Tensor-Namen
func (t Tensor) block() (n int) {
	if _, err := fmt.Sscanf(t.Name, "blk.%d.", &n); err != nil {
		return math.MaxInt
	}
	return
}

// blockSize gibt die Block-Groesse basierend auf dem Tensor-Typ zurueck
func (t Tensor) blockSize() uint64 {
	return TensorType(t.Kind).BlockSize()
}

// BlockSize gibt die Block-Groesse fuer einen TensorType zurueck
// Quantisierte Typen haben typischerweise BlockSize 32 oder 256
func (t TensorType) BlockSize() uint64 {
	switch t {
	case
		TensorTypeF32,
		TensorTypeF16,
		TensorTypeI8,
		TensorTypeI16,
		TensorTypeI32,
		TensorTypeI64,
		TensorTypeF64,
		TensorTypeBF16:
		return 1
	case
		TensorTypeQ4_0,
		TensorTypeQ4_1,
		TensorTypeQ5_0,
		TensorTypeQ5_1,
		TensorTypeQ8_0,
		TensorTypeQ8_1,
		tensorTypeIQ4_NL,
		4, TensorTypeMXFP4:
		return 32
	default:
		return 256
	}
}

// typeSize gibt die Byte-Groesse pro Element zurueck
func (t Tensor) typeSize() uint64 {
	return TensorType(t.Kind).TypeSize()
}

// TypeSize gibt die Byte-Groesse pro Block fuer einen TensorType zurueck
func (t TensorType) TypeSize() uint64 {
	blockSize := t.BlockSize()

	switch t {
	case TensorTypeF32:
		return 4
	case TensorTypeF16:
		return 2
	case TensorTypeQ4_0:
		return 2 + blockSize/2
	case TensorTypeQ4_1:
		return 2 + 2 + blockSize/2
	case TensorTypeQ5_0:
		return 2 + 4 + blockSize/2
	case TensorTypeQ5_1:
		return 2 + 2 + 4 + blockSize/2
	case TensorTypeQ8_0:
		return 2 + blockSize
	case TensorTypeQ8_1:
		return 2 + 2 + blockSize
	case TensorTypeQ2_K:
		return blockSize/16 + blockSize/4 + 2 + 2
	case TensorTypeQ3_K:
		return blockSize/8 + blockSize/4 + 12 + 2
	case TensorTypeQ4_K:
		return 2 + 2 + 12 + blockSize/2
	case TensorTypeQ5_K:
		return 2 + 2 + 12 + blockSize/8 + blockSize/2
	case TensorTypeQ6_K:
		return blockSize/2 + blockSize/4 + blockSize/16 + 2
	case TensorTypeQ8_K:
		return 4 + blockSize + 2*blockSize/16
	case tensorTypeIQ2_XXS:
		return 2 + 2*blockSize/8
	case tensorTypeIQ2_XS:
		return 2 + 2*blockSize/8 + blockSize/32
	case tensorTypeIQ3_XXS:
		return 2 + blockSize/4 + blockSize/8
	case tensorTypeIQ1_S:
		return 2 + blockSize/8 + blockSize/16
	case tensorTypeIQ4_NL:
		return 2 + blockSize/2
	case tensorTypeIQ3_S:
		return 2 + blockSize/4 + blockSize/8 + blockSize/32 + 4
	case tensorTypeIQ2_S:
		return 2 + blockSize/4 + blockSize/16
	case tensorTypeIQ4_XS:
		return 2 + 2 + blockSize/2 + blockSize/64
	case TensorTypeI8:
		return 1
	case TensorTypeI16:
		return 2
	case TensorTypeI32:
		return 4
	case TensorTypeI64:
		return 8
	case TensorTypeF64:
		return 8
	case tensorTypeIQ1_M:
		return blockSize/8 + blockSize/16 + blockSize/32
	case TensorTypeBF16:
		return 2
	case 4, TensorTypeMXFP4:
		return 1 + blockSize/2
	default:
		return 0
	}
}

// Elements gibt die Gesamtanzahl der Elemente im Tensor zurueck
func (t Tensor) Elements() uint64 {
	var count uint64 = 1
	for _, n := range t.Shape {
		count *= n
	}
	return count
}

// Size gibt die Groesse des Tensors in Bytes zurueck
func (t Tensor) Size() uint64 {
	return t.Elements() * t.typeSize() / t.blockSize()
}

// Type gibt den Typ-Namen als String zurueck
func (t Tensor) Type() string {
	return TensorType(t.Kind).String()
}
