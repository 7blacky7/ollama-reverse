// Package ggml - GGUF Container Struktur
//
// Dieses Modul enthaelt die Container-Struktur fuer GGUF-Header:
// - containerGGUF: Repraesentiert den GGUF-Header mit Versionsinformationen
// - Name: Gibt den Container-Namen zurueck
// - Decode: Liest den GGUF-Header und dekodiert das Modell
package ggml

import (
	"encoding/binary"
	"io"
)

// containerGGUF repraesentiert den GGUF-Header mit Versionsinformationen
type containerGGUF struct {
	ByteOrder binary.ByteOrder
	Version   uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}

	V3 struct {
		NumTensor uint64
		NumKV     uint64
	}

	maxArraySize int
}

// Name gibt den Container-Namen zurueck
func (c *containerGGUF) Name() string {
	return "gguf"
}

// Decode liest den GGUF-Header und dekodiert das Modell
func (c *containerGGUF) Decode(rs io.ReadSeeker) (model, error) {
	if err := binary.Read(rs, c.ByteOrder, &c.Version); err != nil {
		return nil, err
	}

	var err error
	switch c.Version {
	case 1:
		err = binary.Read(rs, c.ByteOrder, &c.V1)
	case 2:
		err = binary.Read(rs, c.ByteOrder, &c.V2)
	default:
		err = binary.Read(rs, c.ByteOrder, &c.V3)
	}
	if err != nil {
		return nil, err
	}

	model := newGGUF(c)
	if err := model.Decode(rs); err != nil {
		return nil, err
	}

	return model, nil
}
