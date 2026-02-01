//go:build mlx

// Package mlx - Backend-Initialisierung und Safetensors-Laden
//
// Hauptfunktionen:
// - New: Backend erstellen
// - LoadSafeTensors: Tensor-Dateien laden
// - Get: Tensor nach Name abrufen
// - NewContext: Neuen Kontext erstellen

package mlx

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../../build/_deps/mlx-c-src
#cgo LDFLAGS: -L${SRCDIR}/../../../../build/lib/ollama/ -lmlxc -lmlx
#cgo LDFLAGS: -framework Accelerate
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../../build/lib/ollama/
#include <stdlib.h>
#include "mlx/c/mlx.h"
static inline size_t stride(const mlx_array a, int i) {return mlx_array_strides(a)[i];}

extern void goStackTrace();
static void error_handler(const char *msg, void* data) {
	fprintf(stderr, "MLX error: %s\n", msg);
	goStackTrace();
	exit(-1); // TODO adjust so this can become a return code on the current thread instead of exit
}
static void set_error_handler() {mlx_set_error_handler(&error_handler, NULL, NULL);}
static void* mlx_array_data_float16_asvoid(const mlx_array a) {return (void*)mlx_array_data_float16(a);}
typedef const char cchar_t;
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime/debug"
	"unsafe"

	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/x/ml"
)

func init() {
	ml.RegisterBackend("mlx", New)
	C.set_error_handler()
}

//export goStackTrace
func goStackTrace() {
	debug.PrintStack()
}

// SafetensorsIndexMetadata enthält Metadaten für den Index
type SafetensorsIndexMetadata struct {
	TotalSize uint64 `json:"total_size"`
}

// SafetensorsIndex repräsentiert den Safetensors-Index
type SafetensorsIndex struct {
	Metadata  SafetensorsIndexMetadata `json:"metadata"`
	WeightMap map[string]string        `json:"weight_map"`
}

// Backend implementiert das ML-Backend Interface
type Backend struct {
	meta    fs.Config
	tensors map[string]*Array
}

// New erstellt ein neues MLX-Backend
func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	// TODO assumes modelPath is actually a directory for now...
	kv, tokenizer, err := convert.LoadModelMetadata(os.DirFS(modelPath))
	if err != nil {
		return nil, fmt.Errorf("unable to load model: %w", err)
	}

	b := &Backend{
		meta: kv.KV(tokenizer),
	}

	err = b.LoadSafeTensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("safetensors load failed: %w", err)
	}
	return b, nil
}

// LoadSafeTensors lädt Tensoren aus Safetensor-Dateien
func (b *Backend) LoadSafeTensors(dir string) error {
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("failed to stat dir: %w", err)
	}
	// other variations to try?
	stFilename := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(stFilename); err != nil {
		return fmt.Errorf("failed to stat %s: %w", stFilename, err)
	}

	fp, err := os.Open(stFilename)
	if err != nil {
		return fmt.Errorf("failed to open safetensor index: %s: %w", stFilename, err)
	}
	decoder := json.NewDecoder(fp)
	var index SafetensorsIndex
	if err := decoder.Decode(&index); err != nil {
		return fmt.Errorf("decode error: %s: %w", stFilename, err)
	}
	slog.Info("XXX parsed metadata", "size", index.Metadata.TotalSize, "weights", len(index.WeightMap))
	filenames := map[string]struct{}{}
	for _, filename := range index.WeightMap {
		filenames[filename] = struct{}{}
	}
	stream := C.mlx_default_cpu_stream_new()

	b.tensors = map[string]*Array{}

	for filename := range filenames {
		filepath := filepath.Join(dir, filename)
		if _, err := os.Stat(filepath); err != nil {
			return fmt.Errorf("failed to stat %s: %w", filepath, err)
		}
		slog.Info("Loading tensors from", "filename", filename)
		cFilename := C.CString(filepath)
		defer C.free(unsafe.Pointer(cFilename))
		data := C.mlx_map_string_to_array_new() // TODO is this needed or just var it?
		metadata := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_array_free(data)
		defer C.mlx_map_string_to_string_free(metadata)

		if C.mlx_load_safetensors(&data, &metadata, cFilename, stream) != 0 {
			// TODO with the current error handling, this will never happen
			return fmt.Errorf("load failed")
		}

		it := C.mlx_map_string_to_array_iterator_new(data)
		// 	defer C.mlx_array_free(shaped)
		// TODO confusing how memory management works with this...
		for {
			var key *C.cchar_t
			var value C.mlx_array
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}
			k := C.GoString((*C.char)(key))
			b.tensors[k] = &Array{
				name: k,
				a:    value,
			}
			// slog.Info("XXX read", "tensor", b.tensors[k], "type", b.tensors[k].TypeString())
		}
	}

	return nil
}

// Get gibt einen Tensor nach Name zurück
func (b *Backend) Get(name string) ml.Tensor {
	var t ml.Tensor
	var ok bool
	if t, ok = b.tensors[name]; !ok {
		// slog.Warn("unable to locate", "tensor", name)
		return nil
	}
	// slog.Info("Fetching", "tensor", name, "type", b.tensors[name].TypeString())
	return t
}

// NewContext erstellt einen neuen MLX-Kontext
func (b *Backend) NewContext() ml.Context {
	// slog.Info("MLX.NewContext")
	return &Context{
		stream: C.mlx_default_gpu_stream_new(),
	}
}

// Config gibt die Backend-Konfiguration zurück
func (b *Backend) Config() fs.Config {
	return b.meta
}
