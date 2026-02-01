// Package server - Model Show Handler
// Beinhaltet: ShowHandler, GetModelInfo, getModelData
// Abgetrennt aus routes_models.go fuer bessere Wartbarkeit
package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen"
	xserver "github.com/ollama/ollama/x/server"
)

// ShowHandler verarbeitet /api/show Anfragen
func (s *Server) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	resp, err := GetModelInfo(req)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	c.JSON(http.StatusOK, resp)
}

// GetModelInfo holt detaillierte Model-Informationen
func GetModelInfo(req api.ShowRequest) (*api.ShowResponse, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, model.Unqualified(name)
	}
	name, err := getExistingName(name)
	if err != nil {
		return nil, err
	}

	m, err := GetModel(name.String())
	if err != nil {
		return nil, err
	}

	modelDetails := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.Config.ModelFamily,
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}

	// For image generation models, populate details from imagegen package
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		if info, err := imagegen.GetModelInfo(name.String()); err == nil {
			modelDetails.Family = info.Architecture
			modelDetails.ParameterSize = format.HumanNumber(uint64(info.ParameterCount))
			modelDetails.QuantizationLevel = info.Quantization
		}
	}

	// For safetensors LLM models (experimental), populate details from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			if arch, ok := info["general.architecture"].(string); ok && arch != "" {
				modelDetails.Family = arch
			}
			if paramCount, ok := info["general.parameter_count"].(int64); ok && paramCount > 0 {
				modelDetails.ParameterSize = format.HumanNumber(uint64(paramCount))
			}
		}
		if dtype, err := xserver.GetSafetensorsDtype(name); err == nil && dtype != "" {
			modelDetails.QuantizationLevel = dtype
		}
	}

	if req.System != "" {
		m.System = req.System
	}

	msgs := make([]api.Message, len(m.Messages))
	for i, msg := range m.Messages {
		msgs[i] = api.Message{Role: msg.Role, Content: msg.Content}
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, err
	}

	resp := &api.ShowResponse{
		License:      strings.Join(m.License, "\n"),
		System:       m.System,
		Template:     m.Template.String(),
		Details:      modelDetails,
		Messages:     msgs,
		Capabilities: m.Capabilities(),
		ModifiedAt:   mf.FileInfo().ModTime(),
		Requires:     m.Config.Requires,
		ModelInfo:    make(map[string]any),
	}

	if m.Config.RemoteHost != "" {
		resp.RemoteHost = m.Config.RemoteHost
		resp.RemoteModel = m.Config.RemoteModel

		if m.Config.ModelFamily != "" {
			resp.ModelInfo = make(map[string]any)
			resp.ModelInfo["general.architecture"] = m.Config.ModelFamily

			if m.Config.BaseName != "" {
				resp.ModelInfo["general.basename"] = m.Config.BaseName
			}

			if m.Config.ContextLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.context_length", m.Config.ModelFamily)] = m.Config.ContextLen
			}

			if m.Config.EmbedLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.embedding_length", m.Config.ModelFamily)] = m.Config.EmbedLen
			}
		}
	}

	var params []string
	cs := 30
	for k, v := range m.Options {
		switch val := v.(type) {
		case []any:
			for _, nv := range val {
				params = append(params, fmt.Sprintf("%-*s %#v", cs, k, nv))
			}
		default:
			params = append(params, fmt.Sprintf("%-*s %#v", cs, k, v))
		}
	}
	resp.Parameters = strings.Join(params, "\n")

	if len(req.Options) > 0 {
		if m.Options == nil {
			m.Options = make(map[string]any)
		}
		for k, v := range req.Options {
			m.Options[k] = v
		}
	}

	var sb strings.Builder
	fmt.Fprintln(&sb, "# Modelfile generated by \"ollama show\"")
	fmt.Fprintln(&sb, "# To build a new Modelfile based on this, replace FROM with:")
	fmt.Fprintf(&sb, "# FROM %s\n\n", m.ShortName)
	fmt.Fprint(&sb, m.String())
	resp.Modelfile = sb.String()

	// skip loading tensor information if this is a remote model
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		return resp, nil
	}

	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	// For safetensors LLM models (experimental), populate ModelInfo from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			resp.ModelInfo = info
		}
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	kvData, tensors, err := getModelData(m.ModelPath, req.Verbose)
	if err != nil {
		return nil, err
	}

	delete(kvData, "general.name")
	delete(kvData, "tokenizer.chat_template")
	resp.ModelInfo = kvData

	tensorData := make([]api.Tensor, len(tensors.Items()))
	for cnt, t := range tensors.Items() {
		tensorData[cnt] = api.Tensor{Name: t.Name, Type: t.Type(), Shape: t.Shape}
	}
	resp.Tensors = tensorData

	if len(m.ProjectorPaths) > 0 {
		projectorData, _, err := getModelData(m.ProjectorPaths[0], req.Verbose)
		if err != nil {
			return nil, err
		}
		resp.ProjectorInfo = projectorData
	}

	return resp, nil
}

// getModelData laedt GGML Model-Daten
func getModelData(digest string, verbose bool) (ggml.KV, ggml.Tensors, error) {
	maxArraySize := 0
	if verbose {
		maxArraySize = -1
	}
	data, err := llm.LoadModel(digest, maxArraySize)
	if err != nil {
		return nil, ggml.Tensors{}, err
	}

	kv := data.KV()

	if !verbose {
		for k := range kv {
			if t, ok := kv[k].([]any); len(t) > 5 && ok {
				kv[k] = []any{}
			}
		}
	}

	return kv, data.Tensors(), nil
}
