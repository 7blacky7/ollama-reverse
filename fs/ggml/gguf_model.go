// Package ggml - GGUF Model Struktur und Decode
//
// Dieses Modul enthaelt die Haupt-Modell-Struktur fuer GGUF:
// - gguf: Repraesentiert ein geladenes GGUF-Modell
// - newGGUF: Factory-Funktion fuer gguf-Instanzen
// - KV/Tensors: Zugriffsmethoden
// - numTensor/numKV: Versionsspezifische Getter
// - Decode: Haupt-Deserialisierungsfunktion
// - decodeTensors: Tensor-Metadaten lesen
package ggml

import (
	"fmt"
	"io"
)

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
