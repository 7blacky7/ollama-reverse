// Package ggml - GGUF Write Operations
//
// Dieses Modul enthaelt Funktionen zum Schreiben von GGUF-Dateien:
// - WriteGGUF: Schreibt komplettes GGUF-File mit KV und Tensors
// - writeGGUF: Generische Write-Funktion fuer Basistypen
// - writeGGUFString: String-Serialisierung
// - writeGGUFArray: Array-Serialisierung
// - ggufWriteKV: Key-Value Paar Serialisierung
// - ggufWriteTensorInfo: Tensor-Metadaten Serialisierung
package ggml

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs"
	"golang.org/x/sync/errgroup"
)

// WriteGGUF schreibt ein GGUF-File mit KV-Paaren und Tensors (V3 Format)
func WriteGGUF(f *os.File, kv fs.Config, ts []*Tensor) error {
	arch := kv.String("general.architecture")
	if arch == "" {
		return fmt.Errorf("architecture not set")
	}

	// Magic: "GGUF"
	if err := binary.Write(f, binary.LittleEndian, []byte("GGUF")); err != nil {
		return err
	}

	// Version: 3
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}

	// Tensor Count
	if err := binary.Write(f, binary.LittleEndian, uint64(len(ts))); err != nil {
		return err
	}

	// KV Count
	if err := binary.Write(f, binary.LittleEndian, uint64(kv.Len())); err != nil {
		return err
	}

	// Write KV Pairs
	for _, key := range slices.Sorted(kv.Keys()) {
		if err := ggufWriteKV(f, arch, key, kv.Value(key)); err != nil {
			return err
		}
	}

	// Sort Tensors
	slices.SortStableFunc(ts, func(a, b *Tensor) int {
		return cmp.Or(cmp.Compare(a.block(), b.block()), cmp.Compare(a.Name, b.Name))
	})

	alignment := kv.Uint("general.alignment", 32)

	// Calculate offsets and write tensor info
	var s uint64
	for i := range ts {
		ts[i].Offset = s
		if err := ggufWriteTensorInfo(f, ts[i]); err != nil {
			return err
		}
		s += ts[i].Size()
		s += uint64(ggufPadding(int64(s), int64(alignment)))
	}

	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	offset += ggufPadding(offset, int64(alignment))

	// Write tensor data in parallel
	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, t := range ts {
		w := io.NewOffsetWriter(f, offset+int64(t.Offset))
		g.Go(func() error {
			_, err := t.WriteTo(w)
			return err
		})
	}

	return g.Wait()
}

// writeGGUF schreibt einen typisierten Wert mit Typ-Prefix
func writeGGUF[V any](w io.Writer, t uint32, v V) error {
	if err := binary.Write(w, binary.LittleEndian, t); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, v)
}

// writeGGUFString schreibt einen String mit Typ-Prefix und Laenge
func writeGGUFString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, ggufTypeString); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := io.Copy(w, strings.NewReader(s))
	return err
}

// writeGGUFArray schreibt ein Array mit Typ-Prefix
func writeGGUFArray[S ~[]E, E any](w io.Writer, t uint32, s S) error {
	if err := binary.Write(w, binary.LittleEndian, ggufTypeArray); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, t); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}

	// Strings muessen einzeln geschrieben werden
	if t == ggufTypeString {
		for _, e := range any(s).([]string) {
			if err := binary.Write(w, binary.LittleEndian, uint64(len(e))); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, []byte(e)); err != nil {
				return err
			}
		}
		return nil
	}

	return binary.Write(w, binary.LittleEndian, s)
}

// ggufWriteKV schreibt ein Key-Value Paar
func ggufWriteKV(ws io.WriteSeeker, arch, k string, v any) error {
	// Prefix hinzufuegen falls nicht vorhanden
	if !strings.HasPrefix(k, arch+".") && !strings.HasPrefix(k, "general.") &&
		!strings.HasPrefix(k, "adapter.") && !strings.HasPrefix(k, "tokenizer.") {
		k = arch + "." + k
	}

	slog.Debug(k, "type", fmt.Sprintf("%T", v))

	// Key schreiben
	if err := binary.Write(ws, binary.LittleEndian, uint64(len(k))); err != nil {
		return err
	}
	if err := binary.Write(ws, binary.LittleEndian, []byte(k)); err != nil {
		return err
	}

	// Value schreiben
	var err error
	switch v := v.(type) {
	case int32:
		err = writeGGUF(ws, ggufTypeInt32, v)
	case int64:
		err = writeGGUF(ws, ggufTypeInt64, v)
	case uint32, FileType:
		err = writeGGUF(ws, ggufTypeUint32, v)
	case uint64:
		err = writeGGUF(ws, ggufTypeUint64, v)
	case float32:
		err = writeGGUF(ws, ggufTypeFloat32, v)
	case bool:
		err = writeGGUF(ws, ggufTypeBool, v)
	case string:
		err = writeGGUFString(ws, v)
	case []int32:
		err = writeGGUFArray(ws, ggufTypeInt32, v)
	case *array[int32]:
		err = writeGGUFArray(ws, ggufTypeInt32, v.values)
	case []int64:
		err = writeGGUFArray(ws, ggufTypeInt64, v)
	case *array[int64]:
		err = writeGGUFArray(ws, ggufTypeInt64, v.values)
	case []uint32:
		err = writeGGUFArray(ws, ggufTypeUint32, v)
	case *array[uint32]:
		err = writeGGUFArray(ws, ggufTypeUint32, v.values)
	case []float32:
		err = writeGGUFArray(ws, ggufTypeFloat32, v)
	case *array[float32]:
		err = writeGGUFArray(ws, ggufTypeFloat32, v.values)
	case []string:
		err = writeGGUFArray(ws, ggufTypeString, v)
	case *array[string]:
		err = writeGGUFArray(ws, ggufTypeString, v.values)
	case []bool:
		err = writeGGUFArray(ws, ggufTypeBool, v)
	case *array[bool]:
		err = writeGGUFArray(ws, ggufTypeBool, v.values)
	default:
		return fmt.Errorf("improper type for '%s'", k)
	}
	return err
}

// ggufWriteTensorInfo schreibt die Tensor-Metadaten
func ggufWriteTensorInfo(ws io.WriteSeeker, t *Tensor) error {
	slog.Debug(t.Name, "kind", t.Kind, "shape", t.Shape, "offset", t.Offset)

	// Name
	if err := binary.Write(ws, binary.LittleEndian, uint64(len(t.Name))); err != nil {
		return err
	}
	if err := binary.Write(ws, binary.LittleEndian, []byte(t.Name)); err != nil {
		return err
	}

	// Dimensions
	if err := binary.Write(ws, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
		return err
	}
	for _, n := range t.Shape {
		if err := binary.Write(ws, binary.LittleEndian, n); err != nil {
			return err
		}
	}

	// Kind + Offset
	if err := binary.Write(ws, binary.LittleEndian, t.Kind); err != nil {
		return err
	}
	return binary.Write(ws, binary.LittleEndian, t.Offset)
}

// ggufPadding berechnet das Padding fuer Alignment
func ggufPadding(offset, align int64) int64 {
	return (align - offset%align) % align
}
