// device_layers.go
// Dieses Modul enthaelt die GPU-Layer-Verwaltung fuer die Zuweisung von
// Modell-Layern auf einzelne oder mehrere GPUs.
// Urspruenglich aus device.go extrahiert.

package ml

import (
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"math"
	"slices"
)

// GPULayers is a set of layers to be allocated on a single GPU
type GPULayers struct {
	DeviceID

	// Layers is a set of layer indicies to load
	Layers []int
}

// FirstLayer returns the smallest layer index scheduled on this GPU, or MaxInt when empty.
func (g GPULayers) FirstLayer() int {
	if len(g.Layers) == 0 {
		return math.MaxInt
	}

	first := g.Layers[0]
	for i := 1; i < len(g.Layers); i++ {
		if g.Layers[i] < first {
			first = g.Layers[i]
		}
	}

	return first
}

func (g GPULayers) String() string {
	if len(g.Layers) == 0 {
		return ""
	}

	slices.Sort(g.Layers)

	contiguous := true
	base := g.Layers[0]
	for i := range g.Layers {
		if g.Layers[i] != base+i {
			contiguous = false
			break
		}
	}

	if contiguous {
		return fmt.Sprintf("ID:%v Layers:%v(%v..%v)", g.ID, len(g.Layers), g.Layers[0], g.Layers[len(g.Layers)-1])
	} else {
		return fmt.Sprintf("ID:%v Layers:%v%v", g.ID, len(g.Layers), g.Layers)
	}
}

// GPULayersList is a set of layer allocations across multiple GPUs
type GPULayersList []GPULayers

func (l GPULayersList) Len() int      { return len(l) }
func (l GPULayersList) Swap(i, j int) { l[i], l[j] = l[j], l[i] }

// Sort by the ordering of the layers offloaded
func (l GPULayersList) Less(i, j int) bool {
	li := l[i].FirstLayer()
	lj := l[j].FirstLayer()

	return li < lj
}

func (l GPULayersList) String() string {
	if l.Sum() > 0 {
		return fmt.Sprintf("%v%v", l.Sum(), []GPULayers(l))
	} else {
		return fmt.Sprintf("%v", []GPULayers(l))
	}
}

// Sum is the total number of layers assigned across all GPUs
func (l GPULayersList) Sum() int {
	var sum int

	for _, g := range l {
		sum += len(g.Layers)
	}

	return sum
}

var h maphash.Hash

// Hash is an identifier of this layer assignment
func (l GPULayersList) Hash() uint64 {
	h.Reset()
	for _, g := range l {
		if len(g.Layers) > 0 {
			h.WriteString(g.ID + g.Library)
			for _, l := range g.Layers {
				binary.Write(&h, binary.NativeEndian, int64(l))
			}
		}
	}

	return h.Sum64()
}
