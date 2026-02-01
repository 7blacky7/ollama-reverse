// create_convert.go - Konvertierungs-Funktionen fuer Model-Erstellung
//
// Enthaelt:
// - convertModelFromFiles: Konvertiert Dateien zu Model-Layern
// - detectModelTypeFromFiles: Erkennt Dateityp (safetensors/gguf)
// - convertFromSafetensors: Safetensors zu GGUF Konvertierung
// - kvFromLayers: Extrahiert KV-Config aus Layern
// - quantizeLayer: Quantisiert einen Layer
// - ggufLayers: Erstellt Layer aus GGUF-Datei
// - handleFromModel: Verarbeitet "from" Parameter
// - processInfoFields: Verarbeitet Info-Map
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// convertModelFromFiles konvertiert Dateien basierend auf erkanntem Typ
func convertModelFromFiles(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	switch detectModelTypeFromFiles(files) {
	case "safetensors":
		layers, err := convertFromSafetensors(files, baseLayers, isAdapter, fn)
		if err != nil {
			slog.Error("error converting from safetensors", "error", err)
			return nil, err
		}
		return layers, nil
	case "gguf":
		if len(files) == 0 {
			return nil, errNoFilesProvided
		} else if len(files) > 1 && isAdapter {
			return nil, errOnlyOneAdapterSupported
		}

		var digest string
		var allLayers []*layerGGML
		for _, v := range files {
			digest = v
			layers, err := ggufLayers(digest, fn)
			if err != nil {
				return nil, err
			}
			allLayers = append(allLayers, layers...)
		}
		return allLayers, nil
	default:
		return nil, errUnknownType
	}
}

// detectModelTypeFromFiles erkennt den Dateityp anhand der Dateiendung oder Magic Bytes
func detectModelTypeFromFiles(files map[string]string) string {
	for fn := range files {
		if strings.HasSuffix(fn, ".safetensors") {
			return "safetensors"
		} else if strings.HasSuffix(fn, ".gguf") {
			return "gguf"
		} else {
			// Versuche GGUF anhand der Magic Bytes zu erkennen
			blobPath, err := manifest.BlobsPath(files[fn])
			if err != nil {
				slog.Error("error getting blobs path", "file", fn)
				return ""
			}

			f, err := os.Open(blobPath)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}
			defer f.Close()

			buf := make([]byte, 4)
			_, err = f.Read(buf)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}

			ct := ggml.DetectContentType(buf)
			if ct == "gguf" {
				return "gguf"
			}
		}
	}

	return ""
}

// convertFromSafetensors konvertiert Safetensors-Dateien zu GGUF
func convertFromSafetensors(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	tmpDir, err := os.MkdirTemp(envconfig.Models(), "ollama-safetensors")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	// Root-Verzeichnis fuer Pfadvalidierung
	root, err := os.OpenRoot(tmpDir)
	if err != nil {
		return nil, err
	}
	defer root.Close()

	for fp, digest := range files {
		if !fs.ValidPath(fp) {
			return nil, fmt.Errorf("%w: %s", errFilePath, fp)
		}
		if _, err := root.Stat(fp); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s: %s", errFilePath, err, fp)
		}

		blobPath, err := manifest.BlobsPath(digest)
		if err != nil {
			return nil, err
		}
		if err := createLink(blobPath, filepath.Join(tmpDir, fp)); err != nil {
			return nil, err
		}
	}

	t, err := os.CreateTemp(tmpDir, "fp16")
	if err != nil {
		return nil, err
	}
	defer t.Close()

	var mediaType string
	if !isAdapter {
		fn(api.ProgressResponse{Status: "converting model"})
		mediaType = "application/vnd.ollama.image.model"
		if err := convert.ConvertModel(os.DirFS(tmpDir), t); err != nil {
			return nil, err
		}
	} else {
		kv, err := kvFromLayers(baseLayers)
		if err != nil {
			return nil, err
		}
		fn(api.ProgressResponse{Status: "converting adapter"})
		mediaType = "application/vnd.ollama.image.adapter"
		if err := convert.ConvertAdapter(os.DirFS(tmpDir), t, kv); err != nil {
			return nil, err
		}
	}

	if _, err := t.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(t, mediaType)
	if err != nil {
		return nil, err
	}

	bin, err := layer.Open()
	if err != nil {
		return nil, err
	}
	defer bin.Close()

	f, err := ggml.Decode(bin, -1)
	if err != nil {
		return nil, err
	}
	layers := []*layerGGML{{layer, f}}

	if !isAdapter {
		return detectChatTemplate(layers)
	}
	return layers, nil
}

// kvFromLayers extrahiert KV-Config aus Base-Layern
func kvFromLayers(baseLayers []*layerGGML) (ofs.Config, error) {
	for _, l := range baseLayers {
		if l.GGML != nil {
			return l.KV(), nil
		}
	}
	return ggml.KV{}, fmt.Errorf("no base model was found")
}

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

// handleFromModel verarbeitet "from" Parameter
func handleFromModel(c *gin.Context, r api.CreateRequest, config *model.ConfigV2, fn func(resp api.ProgressResponse)) ([]*layerGGML, bool, error) {
	slog.Debug("create model from model name", "from", r.From)
	fromName := model.ParseName(r.From)
	if !fromName.IsValid() {
		return nil, false, errors.New(errtypes.InvalidModelNameErrMsg)
	}

	if r.RemoteHost != "" {
		ru, err := remoteURL(r.RemoteHost)
		if err != nil {
			return nil, false, errors.New("bad remote")
		}
		config.RemoteModel = r.From
		config.RemoteHost = ru
		return nil, true, nil
	}

	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	baseLayers, err := parseFromModel(ctx, fromName, fn)
	if err != nil {
		return nil, false, err
	}

	// Config-Werte vom Basis-Model uebernehmen
	inheritConfigFromBase(fromName, config)

	return baseLayers, false, nil
}

// inheritConfigFromBase uebernimmt Config-Werte vom Basis-Model
func inheritConfigFromBase(fromName model.Name, config *model.ConfigV2) {
	if config.Renderer != "" && config.Parser != "" && config.Requires != "" {
		return
	}

	mf, mErr := manifest.ParseNamedManifest(fromName)
	if mErr != nil || mf.Config.Digest == "" {
		return
	}

	configPath, pErr := manifest.BlobsPath(mf.Config.Digest)
	if pErr != nil {
		return
	}

	cfgFile, fErr := os.Open(configPath)
	if fErr != nil {
		return
	}
	defer cfgFile.Close()

	var baseConfig model.ConfigV2
	if decErr := json.NewDecoder(cfgFile).Decode(&baseConfig); decErr != nil {
		return
	}

	if config.Renderer == "" {
		config.Renderer = baseConfig.Renderer
	}
	if config.Parser == "" {
		config.Parser = baseConfig.Parser
	}
	if config.Requires == "" {
		config.Requires = baseConfig.Requires
	}
}

// processInfoFields verarbeitet Info-Map und setzt Config-Werte
func processInfoFields(info map[string]any, config *model.ConfigV2) {
	caps, ok := info["capabilities"]
	if ok {
		switch tcaps := caps.(type) {
		case []any:
			caps := make([]string, len(tcaps))
			for i, c := range tcaps {
				str, ok := c.(string)
				if !ok {
					continue
				}
				caps[i] = str
			}
			config.Capabilities = append(config.Capabilities, caps...)
		}
	}

	strFromInfo := func(k string) string {
		v, ok := info[k]
		if ok {
			val := v.(string)
			return val
		}
		return ""
	}

	vFromInfo := func(k string) float64 {
		v, ok := info[k]
		if ok {
			val := v.(float64)
			return val
		}
		return 0
	}

	config.ModelFamily = strFromInfo("model_family")
	if config.ModelFamily != "" {
		config.ModelFamilies = []string{config.ModelFamily}
	}

	config.BaseName = strFromInfo("base_name")
	config.FileType = strFromInfo("quantization_level")
	config.ModelType = strFromInfo("parameter_size")
	config.ContextLen = int(vFromInfo("context_length"))
	config.EmbedLen = int(vFromInfo("embedding_length"))
}
