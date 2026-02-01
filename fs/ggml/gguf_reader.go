// Package ggml - GGUF Reader Funktionen
//
// Dieses Modul enthaelt Low-Level Lese-Funktionen fuer GGUF:
// - readGGUF[T]: Generische Funktion zum Lesen typisierter Werte
// - readGGUFString: String-Deserialisierung (V2+)
// - readGGUFV1String: V1-kompatible String-Deserialisierung (null-terminiert)
// - discardGGUFString: Ueberspringt einen String im Reader
package ggml

import (
	"bytes"
	"encoding/binary"
	"io"
)

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
