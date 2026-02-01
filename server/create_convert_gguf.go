// create_convert_gguf.go - GGUF-Verarbeitung und Quantisierung
//
// Enthaelt:
// - quantizeLayer: Layer-Quantisierung
// - ggufLayers: GGUF-Layer-Erstellung
package server

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
)

// quantizeLayer quantisiert einen Layer auf den angegebenen Typ
func quantizeLayer(layer *layerGGML, quantizeType string, fn func(resp api.ProgressResponse)) (*layerGGML, error) {
	ft := layer.GGML.KV().FileType()
	var doneBytes atomic.Uint64
	totalBytes := uint64(layer.Size) - layer.GGML.Tensors().Offset

	fnWrap := func(n uint64) {
		done := doneBytes.Add(n)
		progress := float32(done) / float32(totalBytes)
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("quantizing %s model to %s", ft, quantizeType),
			Digest:    "0000000000000000000",
			Total:     layer.Size,
			Completed: int64(progress * float32(layer.Size)),
		})
	}

	ftype, err := ggml.ParseFileType(quantizeType)
	if err != nil {
		return nil, err
	}

	blob, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}

	fp, err := os.Open(blob)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	temp, err := os.CreateTemp(filepath.Dir(blob), quantizeType)
	if err != nil {
		return nil, err
	}
	defer temp.Close()
	defer os.Remove(temp.Name())

	if err := quantize(fp, temp, layer.GGML, ftype, fnWrap); err != nil {
		return nil, err
	}

	temp.Seek(0, io.SeekStart)
	fn(api.ProgressResponse{Status: "verifying conversion"})

	newLayer, err := manifest.NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}

	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	f, err := ggml.Decode(temp, 1024)
	if err != nil {
		slog.Error(fmt.Sprintf("error decoding ggml: %s\n", err))
		return nil, err
	}

	return &layerGGML{newLayer, f}, nil
}

// ggufLayers erstellt Layer aus einer GGUF-Datei
func ggufLayers(digest string, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	var layers []*layerGGML

	fn(api.ProgressResponse{Status: "parsing GGUF"})
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		return nil, err
	}

	blob, err := os.Open(blobPath)
	if err != nil {
		return nil, err
	}
	defer blob.Close()

	sr := io.NewSectionReader(blob, 0, 512)
	contentType, err := detectContentType(sr)
	if err != nil {
		return nil, err
	}

	if contentType != "gguf" {
		slog.Error(fmt.Sprintf("unsupported content type: %s", contentType))
		return nil, errOnlyGGUFSupported
	}

	f, err := ggml.Decode(blob, -1)
	if err != nil {
		return nil, err
	}

	mediatype := "application/vnd.ollama.image.model"
	if f.KV().Kind() == "adapter" {
		mediatype = "application/vnd.ollama.image.adapter"
	} else if (f.KV().Uint("block_count") == 0 && f.KV().Uint("vision.block_count") > 0) || f.KV().Kind() == "projector" {
		// Standalone Vision-Model
		mediatype = "application/vnd.ollama.image.projector"
	}

	layer, err := manifest.NewLayerFromLayer(digest, mediatype, blob.Name())
	if err != nil {
		slog.Debug("could not create new layer from layer", "error", err)
		return nil, err
	}

	layers = append(layers, &layerGGML{layer, f})

	return detectChatTemplate(layers)
}
