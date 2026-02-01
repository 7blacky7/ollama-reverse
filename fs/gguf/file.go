// Package gguf - GGUF File Struktur und Open/Close
//
// Dieses Modul enthaelt die File-Hauptstruktur fuer GGUF-Dateien:
// - File: Repraesentiert eine geoeffnete GGUF-Datei
// - Open: Oeffnet und parst eine GGUF-Datei
// - Close: Schliesst die Datei und raeumt Ressourcen auf
// - Type-Konstanten fuer die Datentypen
package gguf

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

// Type-Konstanten fuer GGUF-Datentypen
const (
	typeUint8 uint32 = iota
	typeInt8
	typeUint16
	typeInt16
	typeUint32
	typeInt32
	typeFloat32
	typeBool
	typeString
	typeArray
	typeUint64
	typeInt64
	typeFloat64
)

// ErrUnsupported wird bei nicht unterstuetzten Formaten oder Versionen zurueckgegeben
var ErrUnsupported = errors.New("unsupported")

// File repraesentiert eine geoeffnete GGUF-Datei
type File struct {
	Magic   [4]byte
	Version uint32

	keyValues *lazy[KeyValue]
	tensors   *lazy[TensorInfo]
	offset    int64

	file   *os.File
	reader *bufferedReader
	bts    []byte
}

// Open oeffnet eine GGUF-Datei und parst den Header
func Open(path string) (f *File, err error) {
	f = &File{bts: make([]byte, 4096)}
	f.file, err = os.Open(path)
	if err != nil {
		return nil, err
	}

	f.reader = newBufferedReader(f.file, 32<<10)

	if err := binary.Read(f.reader, binary.LittleEndian, &f.Magic); err != nil {
		return nil, err
	}

	if bytes.Equal(f.Magic[:], []byte("gguf")) {
		return nil, fmt.Errorf("%w file type %v", ErrUnsupported, f.Magic)
	}

	if err := binary.Read(f.reader, binary.LittleEndian, &f.Version); err != nil {
		return nil, err
	}

	if f.Version < 2 {
		return nil, fmt.Errorf("%w version %v", ErrUnsupported, f.Version)
	}

	f.tensors, err = newLazy(f, f.readTensor)
	if err != nil {
		return nil, err
	}

	f.tensors.successFunc = func() error {
		offset := f.reader.offset

		alignment := cmp.Or(f.KeyValue("general.alignment").Int(), 32)
		f.offset = offset + (alignment-offset%alignment)%alignment
		return nil
	}

	f.keyValues, err = newLazy(f, f.readKeyValue)
	if err != nil {
		return nil, err
	}

	return f, nil
}

// Close schliesst die Datei und stoppt alle Lazy-Reader
func (f *File) Close() error {
	f.keyValues.stop()
	f.tensors.stop()
	return f.file.Close()
}
