// create_convert_config.go - Config-Verarbeitung und Model-Handling
//
// Enthaelt:
// - handleFromModel: "from" Parameter Verarbeitung
// - inheritConfigFromBase: Config-Vererbung vom Basis-Model
// - processInfoFields: Info-Map Verarbeitung
package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// handleFromModel verarbeitet "from" Parameter
func handleFromModel(c *gin.Context, r api.CreateRequest, config *model.ConfigV2, fn func(resp api.ProgressResponse)) ([]*layerGGML, bool, error) {
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
