// Package ggml - GGUF Decode Operations
//
// Dieses Modul enthaelt Funktionen zum Lesen von GGUF-Dateien:
// - containerGGUF: Container-Struktur fuer GGUF-Header
// - gguf: Hauptstruktur fuer GGUF-Modelle
// - Decode: Deserialisierung von KV-Paaren und Tensors
// - readGGUF*: Lese-Funktionen fuer verschiedene Datentypen
package ggml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

// GGUF Type Constants
const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
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

// gguf repraesentiert ein geladenes GGUF-Modell
type gguf struct {
	*containerGGUF

	kv      KV
	tensors []*Tensor

	parameters   uint64
	tensorOffset uint64

	scratch [16 << 10]byte
}

// newGGUF erstellt eine neue gguf-Instanz
func newGGUF(container *containerGGUF) *gguf {
	return &gguf{
		containerGGUF: container,
		kv:            make(KV),
	}
}

// KV gibt die Key-Value Paare zurueck
func (llm *gguf) KV() KV {
	return llm.kv
}

// Tensors gibt die Tensor-Liste zurueck
func (llm *gguf) Tensors() Tensors {
	return Tensors{
		items:  llm.tensors,
		Offset: llm.tensorOffset,
	}
}

// numTensor gibt die Tensor-Anzahl zurueck (versionsabhaengig)
func (llm *gguf) numTensor() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumTensor)
	case 2:
		return llm.V2.NumTensor
	default:
		return llm.V3.NumTensor
	}
}

// numKV gibt die KV-Anzahl zurueck (versionsabhaengig)
func (llm *gguf) numKV() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumKV)
	case 2:
		return llm.V2.NumKV
	default:
		return llm.V3.NumKV
	}
}

// Decode liest KV-Paare und Tensors aus dem Reader
func (llm *gguf) Decode(rs io.ReadSeeker) error {
	// KV-Paare dekodieren
	for i := 0; uint64(i) < llm.numKV(); i++ {
		k, err := readGGUFString(llm, rs)
		if err != nil {
			return err
		}

		t, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return err
		}

		var v any
		switch t {
		case ggufTypeUint8:
			v, err = readGGUF[uint8](llm, rs)
		case ggufTypeInt8:
			v, err = readGGUF[int8](llm, rs)
		case ggufTypeUint16:
			v, err = readGGUF[uint16](llm, rs)
		case ggufTypeInt16:
			v, err = readGGUF[int16](llm, rs)
		case ggufTypeUint32:
			v, err = readGGUF[uint32](llm, rs)
		case ggufTypeInt32:
			v, err = readGGUF[int32](llm, rs)
		case ggufTypeUint64:
			v, err = readGGUF[uint64](llm, rs)
		case ggufTypeInt64:
			v, err = readGGUF[int64](llm, rs)
		case ggufTypeFloat32:
			v, err = readGGUF[float32](llm, rs)
		case ggufTypeFloat64:
			v, err = readGGUF[float64](llm, rs)
		case ggufTypeBool:
			v, err = readGGUF[bool](llm, rs)
		case ggufTypeString:
			v, err = readGGUFString(llm, rs)
		case ggufTypeArray:
			v, err = readGGUFArray(llm, rs)
		default:
			return fmt.Errorf("invalid type: %d", t)
		}

		if err != nil {
			return err
		}
		llm.kv[k] = v
	}

	// Tensors dekodieren
	if err := llm.decodeTensors(rs); err != nil {
		return err
	}

	// Parameter-Count als KV hinzufuegen
	llm.kv["general.parameter_count"] = llm.parameters

	// Tensor-Offset berechnen
	alignment := llm.kv.Uint("general.alignment", 32)

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	padding := ggufPadding(offset, int64(alignment))
	llm.tensorOffset = uint64(offset + padding)

	// Tensor-Positionen verifizieren
	for _, tensor := range llm.tensors {
		offset, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return fmt.Errorf("failed to get current offset: %w", err)
		}

		padding := ggufPadding(offset, int64(alignment))
		if _, err := rs.Seek(padding, io.SeekCurrent); err != nil {
			return fmt.Errorf("failed to seek to init padding: %w", err)
		}

		if _, err := rs.Seek(int64(tensor.Size()), io.SeekCurrent); err != nil {
			return fmt.Errorf("failed to seek to tensor: %w", err)
		}
	}

	return nil
}

// decodeTensors liest alle Tensor-Metadaten
func (llm *gguf) decodeTensors(rs io.ReadSeeker) error {
	for range llm.numTensor() {
		name, err := readGGUFString(llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor name: %w", err)
		}

		dims, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor dimensions: %w", err)
		}

		shape := make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			shape[i], err = readGGUF[uint64](llm, rs)
			if err != nil {
				return fmt.Errorf("failed to read tensor shape: %w", err)
			}
		}

		kind, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor kind: %w", err)
		}

		offset, err := readGGUF[uint64](llm, rs)
		if err != nil {
			return fmt.Errorf("failed to read tensor offset: %w", err)
		}

		tensor := Tensor{
			Name:   name,
			Kind:   kind,
			Offset: offset,
			Shape:  shape[:],
		}

		llm.tensors = append(llm.tensors, &tensor)
		llm.parameters += tensor.Elements()
	}
	return nil
}

// readGGUF liest einen typisierten Wert aus dem Reader
func readGGUF[T any](llm *gguf, r io.Reader) (T, error) {
	var t T
	err := binary.Read(r, llm.ByteOrder, &t)
	return t, err
}

// readGGUFV1String liest einen V1-String (null-terminiert)
func readGGUFV1String(llm *gguf, r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, llm.ByteOrder, &length); err != nil {
		return "", err
	}

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(length)); err != nil {
		return "", err
	}

	// V1 Strings sind null-terminiert
	b.Truncate(b.Len() - 1)

	return b.String(), nil
}

// discardGGUFString ueberspringt einen String im Reader
func discardGGUFString(llm *gguf, r io.Reader) error {
	buf := llm.scratch[:8]
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return err
	}

	size := int(llm.ByteOrder.Uint64(buf))
	for size > 0 {
		n, err := r.Read(llm.scratch[:min(size, cap(llm.scratch))])
		if err != nil {
			return err
		}
		size -= n
	}
	return nil
}

// readGGUFString liest einen String aus dem Reader
func readGGUFString(llm *gguf, r io.Reader) (string, error) {
	if llm.Version == 1 {
		return readGGUFV1String(llm, r)
	}

	buf := llm.scratch[:8]
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return "", err
	}

	length := int(llm.ByteOrder.Uint64(buf))
	if length > len(llm.scratch) {
		buf = make([]byte, length)
	} else {
		buf = llm.scratch[:length]
	}
	clear(buf)

	_, err = io.ReadFull(r, buf)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}
