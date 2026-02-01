// llama_core.go
// Kern-Modul: CGo-Bindings, Backend-Initialisierung und Device-Management
// Enthält die grundlegenden C-Bindings und Low-Level-Funktionen

package llama

/*
#cgo CFLAGS: -std=c11
#cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/common
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/vendor
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/tools/mtmd
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/src
#cgo CPPFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include

#include <stdlib.h>
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"

#include "sampling_ext.h"

extern bool llamaProgressCallback(float progress, void *user_data);
extern void llamaLog(int level, char* text, void* user_data);
*/
import "C"

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"runtime/cgo"
	"strings"
	"unsafe"

	_ "github.com/ollama/ollama/llama/llama.cpp/common"
	_ "github.com/ollama/ollama/llama/llama.cpp/src"
	_ "github.com/ollama/ollama/llama/llama.cpp/tools/mtmd"
	_ "github.com/ollama/ollama/llama/llama.cpp/tools/mtmd/models"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

func init() {
	C.llama_log_set(C.ggml_log_callback(C.llamaLog), nil)
}

//export llamaLog
func llamaLog(level C.int, text *C.char, _ unsafe.Pointer) {
	// slog levels zeros INFO and are multiples of 4
	if slog.Default().Enabled(context.TODO(), slog.Level(int(level-C.GGML_LOG_LEVEL_INFO)*4)) {
		fmt.Fprint(os.Stderr, C.GoString(text))
	}
}

func BackendInit() {
	ggml.OnceLoad()
	C.llama_backend_init()
}

type Devices struct {
	ml.DeviceID
	LlamaID uint64
}

func EnumerateGPUs() []Devices {
	var ids []Devices

	for i := range C.ggml_backend_dev_count() {
		device := C.ggml_backend_dev_get(i)

		switch C.ggml_backend_dev_type(device) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			var props C.struct_ggml_backend_dev_props
			C.ggml_backend_dev_get_props(device, &props)
			ids = append(ids, Devices{
				DeviceID: ml.DeviceID{
					ID:      C.GoString(props.id),
					Library: C.GoString(props.library),
				},
				LlamaID: uint64(i),
			})
		}
	}

	return ids
}

func GetModelArch(modelPath string) (string, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))

	gguf_ctx := C.gguf_init_from_file(mp, C.struct_gguf_init_params{no_alloc: true, ctx: (**C.struct_ggml_context)(C.NULL)})
	if gguf_ctx == nil {
		return "", errors.New("unable to load model file")
	}
	defer C.gguf_free(gguf_ctx)

	key := C.CString("general.architecture")
	defer C.free(unsafe.Pointer(key))
	arch_index := C.gguf_find_key(gguf_ctx, key)
	if int(arch_index) < 0 {
		return "", errors.New("unknown model architecture")
	}

	arch := C.gguf_get_val_str(gguf_ctx, arch_index)

	return C.GoString(arch), nil
}

// kvCacheTypeFromStr converts a string cache type to the corresponding GGML type value
func kvCacheTypeFromStr(s string) C.enum_ggml_type {
	if s == "" {
		return C.GGML_TYPE_F16
	}

	switch s {
	case "q8_0":
		return C.GGML_TYPE_Q8_0
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
		return C.GGML_TYPE_F16
	}
}

type ModelParams struct {
	Devices      []uint64
	NumGpuLayers int
	MainGpu      int
	UseMmap      bool
	TensorSplit  []float32
	Progress     func(float32)
	VocabOnly    bool
}

//export llamaProgressCallback
func llamaProgressCallback(progress C.float, userData unsafe.Pointer) C.bool {
	handle := *(*cgo.Handle)(userData)
	callback := handle.Value().(func(float32))
	callback(float32(progress))
	return true
}

func LoadModelFromFile(modelPath string, params ModelParams) (*Model, error) {
	cparams := C.llama_model_default_params()
	cparams.n_gpu_layers = C.int(params.NumGpuLayers)
	cparams.main_gpu = C.int32_t(params.MainGpu)
	cparams.use_mmap = C.bool(params.UseMmap)
	cparams.vocab_only = C.bool(params.VocabOnly)

	var devices []C.ggml_backend_dev_t
	for _, llamaID := range params.Devices {
		devices = append(devices, C.ggml_backend_dev_get(C.size_t(llamaID)))
	}
	if len(devices) > 0 {
		devices = append(devices, C.ggml_backend_dev_t(C.NULL))
		devicesData := &devices[0]

		var devicesPin runtime.Pinner
		devicesPin.Pin(devicesData)
		defer devicesPin.Unpin()

		cparams.devices = devicesData
	}

	if len(params.TensorSplit) > 0 {
		tensorSplitData := &params.TensorSplit[0]

		var tensorSplitPin runtime.Pinner
		tensorSplitPin.Pin(tensorSplitData)
		defer tensorSplitPin.Unpin()

		cparams.tensor_split = (*C.float)(unsafe.Pointer(tensorSplitData))
	}

	if params.Progress != nil {
		handle := cgo.NewHandle(params.Progress)
		defer handle.Delete()

		var handlePin runtime.Pinner
		handlePin.Pin(&handle)
		defer handlePin.Unpin()

		cparams.progress_callback = C.llama_progress_callback(C.llamaProgressCallback)
		cparams.progress_callback_user_data = unsafe.Pointer(&handle)
	}

	m := Model{c: C.llama_model_load_from_file(C.CString(modelPath), cparams)}
	if m.c == nil {
		return nil, fmt.Errorf("unable to load model: %s", modelPath)
	}

	return &m, nil
}

func FreeModel(model *Model) {
	C.llama_model_free(model.c)
}
