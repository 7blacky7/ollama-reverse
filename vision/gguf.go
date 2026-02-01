// MODUL: gguf
// ZWECK: GGUF-Header Parsing fuer Modell-Typ Erkennung
// INPUT: GGUF-Datei (io.ReadSeeker)
// OUTPUT: Modell-Typ als String
// NEBENEFFEKTE: Keine (nur Lesen)
// ABHAENGIGKEITEN: encoding/binary, io (Standard-Library)
// HINWEISE: Liest nur Header und Metadata, nicht die Tensor-Daten

package vision

import (
	"encoding/binary"
	"errors"
	"io"
)

// ============================================================================
// GGUF-Fehler
// ============================================================================

// ErrInvalidGGUF wird zurueckgegeben wenn die GGUF-Datei ungueltig ist.
var ErrInvalidGGUF = errors.New("vision: invalid GGUF file")

// ============================================================================
// GGUF-Konstanten
// ============================================================================

const (
	ggufMagic      = "GGUF"
	ggufTypeString = 8
	ggufTypeArray  = 9
	ggufArchKey    = "general.architecture"
)

// ============================================================================
// detectModelTypeFromGGUF - Modell-Typ aus GGUF erkennen
// ============================================================================

// detectModelTypeFromGGUF liest GGUF-Header und ermittelt den Modell-Typ.
func detectModelTypeFromGGUF(r io.ReadSeeker) (string, error) {
	// GGUF Magic pruefen (4 Bytes)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return "", ErrInvalidGGUF
	}
	if string(magic) != ggufMagic {
		return "", ErrInvalidGGUF
	}

	// Version lesen (4 Bytes, little-endian)
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return "", ErrInvalidGGUF
	}

	// Tensor Count lesen (8 Bytes)
	var tensorCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return "", ErrInvalidGGUF
	}

	// Metadata KV Count lesen (8 Bytes)
	var kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return "", ErrInvalidGGUF
	}

	// Metadata durchsuchen
	return searchArchitectureKey(r, kvCount)
}

// ============================================================================
// searchArchitectureKey - Architektur-Key in Metadata finden
// ============================================================================

// searchArchitectureKey durchsucht Metadata nach dem Architektur-Key.
func searchArchitectureKey(r io.ReadSeeker, kvCount uint64) (string, error) {
	for i := uint64(0); i < kvCount; i++ {
		// Key-Laenge lesen (8 Bytes)
		var keyLen uint64
		if err := binary.Read(r, binary.LittleEndian, &keyLen); err != nil {
			return "", ErrInvalidGGUF
		}

		// Key lesen
		key := make([]byte, keyLen)
		if _, err := io.ReadFull(r, key); err != nil {
			return "", ErrInvalidGGUF
		}

		// Value-Type lesen (4 Bytes)
		var valueType uint32
		if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
			return "", ErrInvalidGGUF
		}

		// Pruefe ob dies der Architektur-Key ist
		if string(key) == ggufArchKey {
			return readGGUFString(r, valueType)
		}

		// Value ueberspringen
		if err := skipGGUFValue(r, valueType); err != nil {
			return "", err
		}
	}

	return "", ErrUnknownModelType
}

// ============================================================================
// readGGUFString - GGUF String-Wert lesen
// ============================================================================

// readGGUFString liest einen String-Wert aus GGUF.
func readGGUFString(r io.Reader, valueType uint32) (string, error) {
	if valueType != ggufTypeString {
		return "", ErrInvalidGGUF
	}

	// String-Laenge lesen (8 Bytes)
	var strLen uint64
	if err := binary.Read(r, binary.LittleEndian, &strLen); err != nil {
		return "", err
	}

	// String lesen
	str := make([]byte, strLen)
	if _, err := io.ReadFull(r, str); err != nil {
		return "", err
	}

	return string(str), nil
}

// ============================================================================
// skipGGUFValue - GGUF-Wert ueberspringen
// ============================================================================

// skipGGUFValue ueberspringt einen GGUF-Wert basierend auf seinem Typ.
func skipGGUFValue(r io.ReadSeeker, valueType uint32) error {
	// Byte-Groessen nach GGUF Value Type
	typeSizes := map[uint32]int64{
		0:  1, // UINT8
		1:  1, // INT8
		2:  2, // UINT16
		3:  2, // INT16
		4:  4, // UINT32
		5:  4, // INT32
		6:  4, // FLOAT32
		7:  1, // BOOL
		10: 8, // UINT64
		11: 8, // INT64
		12: 8, // FLOAT64
	}

	// String hat variable Laenge
	if valueType == ggufTypeString {
		var strLen uint64
		if err := binary.Read(r, binary.LittleEndian, &strLen); err != nil {
			return err
		}
		_, err := r.Seek(int64(strLen), io.SeekCurrent)
		return err
	}

	// Array-Typen benoetigen spezielle Behandlung
	if valueType == ggufTypeArray {
		return skipGGUFArray(r)
	}

	// Feste Groesse
	size, ok := typeSizes[valueType]
	if !ok {
		return ErrInvalidGGUF
	}

	_, err := r.Seek(size, io.SeekCurrent)
	return err
}

// ============================================================================
// skipGGUFArray - GGUF-Array ueberspringen
// ============================================================================

// skipGGUFArray ueberspringt ein GGUF-Array.
func skipGGUFArray(r io.ReadSeeker) error {
	// Element-Type lesen (4 Bytes)
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return err
	}

	// Element-Count lesen (8 Bytes)
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return err
	}

	// Alle Elemente ueberspringen
	for i := uint64(0); i < count; i++ {
		if err := skipGGUFValue(r, elemType); err != nil {
			return err
		}
	}

	return nil
}
