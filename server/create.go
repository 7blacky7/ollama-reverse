// create.go - HTTP-Handler fuer Model-Erstellung
//
// Enthaelt:
// - CreateHandler: HTTP-Handler fuer /api/create
// - remoteURL: Parsed Remote-Host URLs
//
// Verwandte Dateien:
// - create_convert.go: Konvertierungs-Funktionen (safetensors, gguf)
// - create_layers.go: Layer-Management und createModel
package server

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

var (
	errNoFilesProvided         = errors.New("no files provided to convert")
	errOnlyOneAdapterSupported = errors.New("only one adapter is currently supported")
	errOnlyGGUFSupported       = errors.New("supplied file was not in GGUF format")
	errUnknownType             = errors.New("unknown type")
	errNeitherFromOrFiles      = errors.New("neither 'from' or 'files' was specified")
	errFilePath                = errors.New("file path must be relative")
)

// CreateHandler verarbeitet POST /api/create Requests
func (s *Server) CreateHandler(c *gin.Context) {
	config := &model.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: model.RootFS{
			Type: "layers",
		},
	}

	var r api.CreateRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	config.Renderer = r.Renderer
	config.Parser = r.Parser
	config.Requires = r.Requires

	for v := range r.Files {
		if !fs.ValidPath(v) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errFilePath.Error()})
			return
		}
	}

	name := model.ParseName(cmp.Or(r.Model, r.Name))
	if !name.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
		return
	}

	name, err := getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(resp api.ProgressResponse) {
			ch <- resp
		}

		oldManifest, _ := manifest.ParseNamedManifest(name)

		var baseLayers []*layerGGML
		var err error
		var remote bool

		if r.From != "" {
			baseLayers, remote, err = handleFromModel(c, r, config, fn)
			if err != nil {
				ch <- gin.H{"error": err.Error()}
				return
			}
		} else if r.Files != nil {
			baseLayers, err = convertModelFromFiles(r.Files, baseLayers, false, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errUnknownType} {
					if errors.Is(err, badReq) {
						ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
						return
					}
				}
				ch <- gin.H{"error": err.Error()}
				return
			}
		} else {
			ch <- gin.H{"error": errNeitherFromOrFiles.Error(), "status": http.StatusBadRequest}
			return
		}

		// Adapter verarbeiten
		var adapterLayers []*layerGGML
		if !remote && r.Adapters != nil {
			adapterLayers, err = convertModelFromFiles(r.Adapters, baseLayers, true, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyOneAdapterSupported, errOnlyGGUFSupported, errUnknownType, errFilePath} {
					if errors.Is(err, badReq) {
						ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
						return
					}
				}
				ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
				return
			}
		}

		if len(adapterLayers) > 0 {
			baseLayers = append(baseLayers, adapterLayers...)
		}

		// Info-Felder verarbeiten
		if r.Info != nil {
			processInfoFields(r.Info, config)
		}

		if err := createModel(r, name, baseLayers, config, fn); err != nil {
			if errors.Is(err, errBadTemplate) {
				ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
				return
			}
			ch <- gin.H{"error": err.Error()}
			return
		}

		if !envconfig.NoPrune() && oldManifest != nil {
			if err := oldManifest.RemoveLayers(); err != nil {
				ch <- gin.H{"error": err.Error()}
			}
		}

		ch <- api.ProgressResponse{Status: "success"}
	}()

	if r.Stream != nil && !*r.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

// remoteURL parsed und normalisiert Remote-Host URLs
func remoteURL(raw string) (string, error) {
	// Spezialfall: Nur Pfad angegeben ("/foo/bar")
	if strings.HasPrefix(raw, "/") {
		return (&url.URL{
			Scheme: "http",
			Host:   net.JoinHostPort("localhost", "11434"),
			Path:   path.Clean(raw),
		}).String(), nil
	}

	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}

	if raw == "ollama.com" || raw == "http://ollama.com" {
		raw = "https://ollama.com:443"
	}

	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	if u.Host == "" {
		u.Host = "localhost"
	}

	hostPart, portPart, err := net.SplitHostPort(u.Host)
	if err == nil {
		u.Host = net.JoinHostPort(hostPart, portPart)
	} else {
		u.Host = net.JoinHostPort(u.Host, "11434")
	}

	if u.Path != "" {
		u.Path = path.Clean(u.Path)
	}

	if u.Path == "/" {
		u.Path = ""
	}

	return u.String(), nil
}
