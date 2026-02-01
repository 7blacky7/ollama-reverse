// Package ggml - GGUF Array Handling
//
// Dieses Modul enthaelt Array-spezifische Datenstrukturen und Funktionen:
// - array[T]: Generische Array-Struktur mit Groessenlimit
// - newArray: Factory-Funktion fuer Arrays
// - readGGUFArray: Array-Deserialisierung
// - readGGUFArrayData: Typisierte Array-Daten lesen
// - readGGUFStringsData: String-Array Deserialisierung
// - readGGUFV1StringsData: V1-kompatible String-Array Deserialisierung
package ggml

import (
	"encoding/json"
	"fmt"
	"io"
)

// array ist eine generische Array-Struktur mit optionalem Groessenlimit
// Bei Arrays groesser als maxSize wird nur die Groesse gespeichert
type array[T any] struct {
	// size ist die tatsaechliche Groesse des Arrays
	size int

	// values enthaelt die Array-Werte. nil wenn Array groesser als maxSize
	values []T
}

// MarshalJSON serialisiert das Array als JSON
func (a *array[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.values)
}

// newArray erstellt ein neues Array mit optionalem Groessenlimit
// Bei maxSize < 0 oder size <= maxSize werden values allokiert
func newArray[T any](size, maxSize int) *array[T] {
	a := array[T]{size: size}
	if maxSize < 0 || size <= maxSize {
		a.values = make([]T, size)
	}
	return &a
}

// readGGUFArray liest ein typisiertes Array aus dem Reader
func readGGUFArray(llm *gguf, r io.Reader) (any, error) {
	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint64](llm, r)
	if err != nil {
		return nil, err
	}

	switch t {
	case ggufTypeUint8:
		return readGGUFArrayData(llm, r, newArray[uint8](int(n), llm.maxArraySize))
	case ggufTypeInt8:
		return readGGUFArrayData(llm, r, newArray[int8](int(n), llm.maxArraySize))
	case ggufTypeUint16:
		return readGGUFArrayData(llm, r, newArray[uint16](int(n), llm.maxArraySize))
	case ggufTypeInt16:
		return readGGUFArrayData(llm, r, newArray[int16](int(n), llm.maxArraySize))
	case ggufTypeUint32:
		return readGGUFArrayData(llm, r, newArray[uint32](int(n), llm.maxArraySize))
	case ggufTypeInt32:
		return readGGUFArrayData(llm, r, newArray[int32](int(n), llm.maxArraySize))
	case ggufTypeUint64:
		return readGGUFArrayData(llm, r, newArray[uint64](int(n), llm.maxArraySize))
	case ggufTypeInt64:
		return readGGUFArrayData(llm, r, newArray[int64](int(n), llm.maxArraySize))
	case ggufTypeFloat32:
		return readGGUFArrayData(llm, r, newArray[float32](int(n), llm.maxArraySize))
	case ggufTypeFloat64:
		return readGGUFArrayData(llm, r, newArray[float64](int(n), llm.maxArraySize))
	case ggufTypeBool:
		return readGGUFArrayData(llm, r, newArray[bool](int(n), llm.maxArraySize))
	case ggufTypeString:
		a := newArray[string](int(n), llm.maxArraySize)
		if llm.Version == 1 {
			return readGGUFV1StringsData(llm, r, a)
		}
		return readGGUFStringsData(llm, r, a)
	default:
		return nil, fmt.Errorf("invalid array type: %d", t)
	}
}

// readGGUFArrayData liest typisierte Array-Daten
func readGGUFArrayData[T any](llm *gguf, r io.Reader, a *array[T]) (any, error) {
	for i := range a.size {
		e, err := readGGUF[T](llm, r)
		if err != nil {
			return nil, err
		}
		if a.values != nil {
			a.values[i] = e
		}
	}
	return a, nil
}

// readGGUFStringsData liest ein String-Array (V2+)
func readGGUFStringsData(llm *gguf, r io.Reader, a *array[string]) (any, error) {
	for i := range a.size {
		if a.values != nil {
			e, err := readGGUFString(llm, r)
			if err != nil {
				return nil, err
			}
			a.values[i] = e
		} else {
			discardGGUFString(llm, r)
		}
	}
	return a, nil
}

// readGGUFV1StringsData liest ein String-Array (V1, null-terminiert)
func readGGUFV1StringsData(llm *gguf, r io.Reader, a *array[string]) (any, error) {
	for i := range a.size {
		if a.values != nil {
			e, err := readGGUFV1String(llm, r)
			if err != nil {
				return nil, err
			}
			a.values[i] = e
		} else {
			_ = discardGGUFString(llm, r)
		}
	}
	return a, nil
}
