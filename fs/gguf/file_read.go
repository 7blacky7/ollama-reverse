// Package gguf - GGUF File Read Funktionen
//
// Dieses Modul enthaelt die Low-Level Lese-Funktionen fuer GGUF-Dateien:
// - readTensor: Liest Tensor-Metadaten
// - readKeyValue: Liest ein Key-Value Paar
// - read[T]: Generische Funktion zum Lesen typisierter Werte
// - readString: String-Deserialisierung
// - readArray: Array-Deserialisierung mit Typ-Erkennung
// - readArrayData[T]: Generische Array-Daten lesen
// - readArrayString: String-Array lesen
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
)

// readTensor liest die Metadaten eines einzelnen Tensors
func (f *File) readTensor() (TensorInfo, error) {
	name, err := readString(f)
	if err != nil {
		return TensorInfo{}, err
	}

	dims, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	shape := make([]uint64, dims)
	for i := range dims {
		shape[i], err = read[uint64](f)
		if err != nil {
			return TensorInfo{}, err
		}
	}

	type_, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	offset, err := read[uint64](f)
	if err != nil {
		return TensorInfo{}, err
	}

	return TensorInfo{
		Name:   name,
		Offset: offset,
		Shape:  shape,
		Type:   TensorType(type_),
	}, nil
}

// readKeyValue liest ein einzelnes Key-Value Paar
func (f *File) readKeyValue() (KeyValue, error) {
	key, err := readString(f)
	if err != nil {
		return KeyValue{}, err
	}

	t, err := read[uint32](f)
	if err != nil {
		return KeyValue{}, err
	}

	value, err := func() (any, error) {
		switch t {
		case typeUint8:
			return read[uint8](f)
		case typeInt8:
			return read[int8](f)
		case typeUint16:
			return read[uint16](f)
		case typeInt16:
			return read[int16](f)
		case typeUint32:
			return read[uint32](f)
		case typeInt32:
			return read[int32](f)
		case typeUint64:
			return read[uint64](f)
		case typeInt64:
			return read[int64](f)
		case typeFloat32:
			return read[float32](f)
		case typeFloat64:
			return read[float64](f)
		case typeBool:
			return read[bool](f)
		case typeString:
			return readString(f)
		case typeArray:
			return readArray(f)
		default:
			return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
		}
	}()
	if err != nil {
		return KeyValue{}, err
	}

	return KeyValue{
		Key:   key,
		Value: Value{value},
	}, nil
}

// read liest einen typisierten Wert aus dem Reader
func read[T any](f *File) (t T, err error) {
	err = binary.Read(f.reader, binary.LittleEndian, &t)
	return t, err
}

// readString liest einen String aus dem Reader
func readString(f *File) (string, error) {
	n, err := read[uint64](f)
	if err != nil {
		return "", err
	}

	if int(n) > len(f.bts) {
		f.bts = make([]byte, n)
	}

	bts := f.bts[:n]
	if _, err := io.ReadFull(f.reader, bts); err != nil {
		return "", err
	}
	defer clear(bts)

	return string(bts), nil
}

// readArray liest ein typisiertes Array aus dem Reader
func readArray(f *File) (any, error) {
	t, err := read[uint32](f)
	if err != nil {
		return nil, err
	}

	n, err := read[uint64](f)
	if err != nil {
		return nil, err
	}

	switch t {
	case typeUint8:
		return readArrayData[uint8](f, n)
	case typeInt8:
		return readArrayData[int8](f, n)
	case typeUint16:
		return readArrayData[uint16](f, n)
	case typeInt16:
		return readArrayData[int16](f, n)
	case typeUint32:
		return readArrayData[uint32](f, n)
	case typeInt32:
		return readArrayData[int32](f, n)
	case typeUint64:
		return readArrayData[uint64](f, n)
	case typeInt64:
		return readArrayData[int64](f, n)
	case typeFloat32:
		return readArrayData[float32](f, n)
	case typeFloat64:
		return readArrayData[float64](f, n)
	case typeBool:
		return readArrayData[bool](f, n)
	case typeString:
		return readArrayString(f, n)
	default:
		return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
	}
}

// readArrayData liest typisierte Array-Daten
func readArrayData[T any](f *File, n uint64) (s []T, err error) {
	s = make([]T, n)
	for i := range n {
		e, err := read[T](f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

// readArrayString liest ein String-Array
func readArrayString(f *File, n uint64) (s []string, err error) {
	s = make([]string, n)
	for i := range n {
		e, err := readString(f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}
