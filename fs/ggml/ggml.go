// Package ggml - Core Types und Interface
//
// Dieses Modul definiert die Kernstrukturen:
// - GGML: Hauptcontainer fuer GGML-Modelle
// - container: Interface fuer verschiedene GGML-Formate
// - model: Interface fuer Model-Daten (KV + Tensors)
// - Decode: Laedt ein GGML-Modell aus einem Reader
// - Magic Constants: File-Format Erkennung
package ggml

import (
	"encoding/binary"
	"errors"
	"io"

	"github.com/ollama/ollama/fs/util/bufioutil"
)

// GGML repraesentiert ein geladenes GGML-Modell
type GGML struct {
	container
	model
	Length int64
}

// model definiert das Interface fuer GGML-Model-Daten
type model interface {
	KV() KV
	Tensors() Tensors
}

// container definiert das Interface fuer GGML-Container-Formate
type container interface {
	Name() string
	Decode(io.ReadSeeker) (model, error)
}

// Magic Constants fuer GGML File-Formate
const (
	// FILE_MAGIC_GGML fuer unversionierte ggml-Dateien
	FILE_MAGIC_GGML = 0x67676d6c
	// FILE_MAGIC_GGMF fuer versionierte ggmf-Dateien
	FILE_MAGIC_GGMF = 0x67676d66
	// FILE_MAGIC_GGJT fuer versionierte ggjt-Dateien
	FILE_MAGIC_GGJT = 0x67676a74
	// FILE_MAGIC_GGLA fuer LoRA-Adapter
	FILE_MAGIC_GGLA = 0x67676C61
	// FILE_MAGIC_GGUF_LE fuer GGUF Little-Endian
	FILE_MAGIC_GGUF_LE = 0x46554747
	// FILE_MAGIC_GGUF_BE fuer GGUF Big-Endian
	FILE_MAGIC_GGUF_BE = 0x47475546
)

// ErrUnsupportedFormat wird zurueckgegeben wenn das Format nicht unterstuetzt wird
var ErrUnsupportedFormat = errors.New("unsupported model format")

// DetectContentType erkennt das GGML-Format anhand der Magic-Bytes
func DetectContentType(b []byte) string {
	switch binary.LittleEndian.Uint32(b[:4]) {
	case FILE_MAGIC_GGML:
		return "ggml"
	case FILE_MAGIC_GGMF:
		return "ggmf"
	case FILE_MAGIC_GGJT:
		return "ggjt"
	case FILE_MAGIC_GGLA:
		return "ggla"
	case FILE_MAGIC_GGUF_LE, FILE_MAGIC_GGUF_BE:
		return "gguf"
	default:
		return ""
	}
}

// Decode dekodiert ein GGML-Modell aus dem Reader.
//
// maxArraySize bestimmt die maximale Array-Groesse fuer KV-Werte.
// Bei negativem Wert werden alle Arrays gesammelt.
func Decode(rs io.ReadSeeker, maxArraySize int) (*GGML, error) {
	rs = bufioutil.NewBufferedSeeker(rs, 32<<10)

	var magic uint32
	if err := binary.Read(rs, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGUF_LE:
		c = &containerGGUF{ByteOrder: binary.LittleEndian, maxArraySize: maxArraySize}
	case FILE_MAGIC_GGUF_BE:
		c = &containerGGUF{ByteOrder: binary.BigEndian, maxArraySize: maxArraySize}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := c.Decode(rs)
	if err != nil {
		return nil, err
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	return &GGML{
		container: c,
		model:     model,
		Length:    offset,
	}, nil
}
