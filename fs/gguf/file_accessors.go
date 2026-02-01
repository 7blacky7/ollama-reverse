// Package gguf - GGUF File Accessor Methoden
//
// Dieses Modul enthaelt die Zugriffs-Methoden fuer GGUF-Dateien:
// - KeyValue: Sucht ein Key-Value Paar nach Name
// - NumKeyValues: Gibt die Anzahl der KV-Paare zurueck
// - KeyValues: Iterator ueber alle KV-Paare
// - TensorInfo: Sucht Tensor-Info nach Name
// - NumTensors: Gibt die Anzahl der Tensors zurueck
// - TensorInfos: Iterator ueber alle Tensor-Infos
// - TensorReader: Liefert einen Reader fuer Tensor-Daten
package gguf

import (
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
)

// KeyValue sucht ein Key-Value Paar nach Name
// Wenn der Key nicht mit "general." oder "tokenizer." beginnt,
// wird der Architecture-Prefix automatisch hinzugefuegt
func (f *File) KeyValue(key string) KeyValue {
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		key = f.KeyValue("general.architecture").String() + "." + key
	}

	if index := slices.IndexFunc(f.keyValues.values, func(kv KeyValue) bool {
		return kv.Key == key
	}); index >= 0 {
		return f.keyValues.values[index]
	}

	for keyValue, ok := f.keyValues.next(); ok; keyValue, ok = f.keyValues.next() {
		if keyValue.Key == key {
			return keyValue
		}
	}

	return KeyValue{}
}

// NumKeyValues gibt die Anzahl der Key-Value Paare zurueck
func (f *File) NumKeyValues() int {
	return int(f.keyValues.count)
}

// KeyValues gibt einen Iterator ueber alle Key-Value Paare zurueck
func (f *File) KeyValues() iter.Seq2[int, KeyValue] {
	return f.keyValues.All()
}

// TensorInfo sucht Tensor-Info nach Name
func (f *File) TensorInfo(name string) TensorInfo {
	if index := slices.IndexFunc(f.tensors.values, func(t TensorInfo) bool {
		return t.Name == name
	}); index >= 0 {
		return f.tensors.values[index]
	}

	// Fast-forward durch KeyValues falls noch nicht geschehen
	_ = f.keyValues.rest()
	for tensor, ok := f.tensors.next(); ok; tensor, ok = f.tensors.next() {
		if tensor.Name == name {
			return tensor
		}
	}

	return TensorInfo{}
}

// NumTensors gibt die Anzahl der Tensors zurueck
func (f *File) NumTensors() int {
	return int(f.tensors.count)
}

// TensorInfos gibt einen Iterator ueber alle Tensor-Infos zurueck
func (f *File) TensorInfos() iter.Seq2[int, TensorInfo] {
	// Fast-forward durch KeyValues falls noch nicht geschehen
	f.keyValues.rest()
	return f.tensors.All()
}

// TensorReader liefert Tensor-Info und einen Reader fuer die Tensor-Daten
func (f *File) TensorReader(name string) (TensorInfo, io.Reader, error) {
	t := f.TensorInfo(name)
	if t.NumBytes() == 0 {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}

	// Fast-forward durch Tensor-Infos falls noch nicht geschehen
	_ = f.tensors.rest()
	return t, io.NewSectionReader(f.file, f.offset+int64(t.Offset), t.NumBytes()), nil
}
