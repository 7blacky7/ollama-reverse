// client.go - HuggingFace Hub Client Wrapper
// Stellt einen HTTP-Client fuer den HuggingFace Hub bereit.
// Autor: Agent 1 - Phase 9
package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Konstanten fuer HuggingFace Hub API
const (
	DefaultHubURL        = "https://huggingface.co"
	DefaultAPIURL        = "https://huggingface.co/api"
	DefaultClientTimeout = 1800 // 30 Minuten fuer grosse Model-Downloads
	EnvHFToken           = "HF_TOKEN"
	EnvHFHome            = "HF_HOME"
	EnvHFEndpoint        = "HF_ENDPOINT"
	ClientUserAgent      = "ollama-reverse/1.0"
)

// Fehler-Definitionen
var (
	ErrModelNotFound   = errors.New("modell nicht gefunden")
	ErrUnauthorized    = errors.New("authentifizierung fehlgeschlagen")
	ErrRateLimited     = errors.New("rate limit ueberschritten")
	ErrNetworkError    = errors.New("netzwerkfehler")
	ErrInvalidModelID  = errors.New("ungueltige modell-id")
	ErrFileNotFound    = errors.New("datei nicht gefunden")
	ErrDownloadFailed  = errors.New("download fehlgeschlagen")
	ErrInvalidResponse = errors.New("ungueltige server-antwort")
)

// APIModelInfo enthaelt Metadaten eines HuggingFace Modells aus der API
type APIModelInfo struct {
	ID           string       `json:"id"`
	ModelID      string       `json:"modelId"`
	Author       string       `json:"author"`
	SHA          string       `json:"sha"`
	LastModified time.Time    `json:"lastModified"`
	Private      bool         `json:"private"`
	Gated        interface{}  `json:"gated"` // Kann bool oder string sein (false, "auto", "manual")
	Pipeline     string       `json:"pipeline_tag"`
	Tags         []string     `json:"tags"`
	Downloads    int64        `json:"downloads"`
	Likes        int64        `json:"likes"`
	LibraryName  string       `json:"library_name"`
	Siblings     []APISibling `json:"siblings"`
}

// IsGated prueft ob das Modell gated ist (authentifizierung erforderlich)
func (m *APIModelInfo) IsGated() bool {
	switch v := m.Gated.(type) {
	case bool:
		return v
	case string:
		return v == "auto" || v == "manual"
	default:
		return false
	}
}

// APISibling repraesentiert eine Datei im Model-Repository
type APISibling struct {
	Filename string   `json:"rfilename"`
	Size     int64    `json:"size"`
	BlobID   string   `json:"blobId"`
	LFS      *LFSInfo `json:"lfs,omitempty"`
}

// LFSInfo enthaelt LFS-Metadaten fuer grosse Dateien
type LFSInfo struct {
	Size        int64  `json:"size"`
	SHA256      string `json:"sha256"`
	PointerSize int64  `json:"pointerSize"`
}

// Client ist der HuggingFace Hub Client
type Client struct {
	httpClient *http.Client
	baseURL    string
	apiURL     string
	token      string
	userAgent  string
}

// ClientOption ist eine Funktion zur Konfiguration des Clients
type ClientOption func(*Client)

// WithToken setzt den HuggingFace API Token
func WithToken(token string) ClientOption {
	return func(c *Client) { c.token = token }
}

// WithBaseURL setzt eine Custom Base-URL
func WithBaseURL(url string) ClientOption {
	return func(c *Client) {
		c.baseURL = strings.TrimSuffix(url, "/")
		c.apiURL = c.baseURL + "/api"
	}
}

// WithClientTimeout setzt den HTTP Timeout
func WithClientTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) { c.httpClient.Timeout = timeout }
}

// WithUserAgent setzt einen Custom User-Agent
func WithUserAgent(ua string) ClientOption {
	return func(c *Client) { c.userAgent = ua }
}

// WithHTTPClient setzt einen Custom HTTP Client
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *Client) { c.httpClient = client }
}

// NewClient erstellt einen neuen HuggingFace Hub Client
func NewClient(options ...ClientOption) *Client {
	c := &Client{
		httpClient: &http.Client{Timeout: DefaultClientTimeout * time.Second},
		baseURL:    DefaultHubURL,
		apiURL:     DefaultAPIURL,
		userAgent:  ClientUserAgent,
	}
	if token := os.Getenv(EnvHFToken); token != "" {
		c.token = token
	}
	if endpoint := os.Getenv(EnvHFEndpoint); endpoint != "" {
		c.baseURL = strings.TrimSuffix(endpoint, "/")
		c.apiURL = c.baseURL + "/api"
	}
	for _, opt := range options {
		opt(c)
	}
	return c
}

// GetModelInfo ruft Metadaten eines Modells ab
func (c *Client) GetModelInfo(modelID string) (*APIModelInfo, error) {
	return c.GetModelInfoWithContext(context.Background(), modelID)
}

// GetModelInfoWithContext ruft Metadaten eines Modells mit Context ab
func (c *Client) GetModelInfoWithContext(ctx context.Context, modelID string) (*APIModelInfo, error) {
	if err := validateModelID(modelID); err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/models/%s", c.apiURL, modelID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNetworkError, err)
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNetworkError, err)
	}
	defer resp.Body.Close()
	if err := c.handleResponseError(resp); err != nil {
		return nil, err
	}
	var info APIModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	return &info, nil
}

// DownloadFile laedt eine einzelne Datei aus einem Model-Repository herunter
func (c *Client) DownloadFile(modelID, filename string) (string, error) {
	return c.DownloadFileWithContext(context.Background(), modelID, filename, "main")
}

// DownloadFileWithRevision laedt eine Datei mit spezifischer Revision herunter
func (c *Client) DownloadFileWithRevision(modelID, filename, revision string) (string, error) {
	return c.DownloadFileWithContext(context.Background(), modelID, filename, revision)
}

// DownloadFileWithContext laedt eine Datei mit Context und Revision herunter
func (c *Client) DownloadFileWithContext(ctx context.Context, modelID, filename, revision string) (string, error) {
	if err := validateModelID(modelID); err != nil {
		return "", err
	}
	if filename == "" {
		return "", fmt.Errorf("%w: dateiname darf nicht leer sein", ErrFileNotFound)
	}
	if revision == "" {
		revision = "main"
	}
	// Cache-Pfad ermitteln
	cacheDir := GetCacheDir()
	modelCacheDir := filepath.Join(cacheDir, "models--"+strings.ReplaceAll(modelID, "/", "--"))
	snapshotDir := filepath.Join(modelCacheDir, "snapshots", revision)
	targetPath := filepath.Join(snapshotDir, filename)
	// Pruefen ob bereits im Cache
	if _, err := os.Stat(targetPath); err == nil {
		return targetPath, nil
	}
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return "", fmt.Errorf("verzeichnis erstellen fehlgeschlagen: %w", err)
	}
	// Download
	url := fmt.Sprintf("%s/%s/resolve/%s/%s", c.baseURL, modelID, revision, filename)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrNetworkError, err)
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrNetworkError, err)
	}
	defer resp.Body.Close()
	if err := c.handleResponseError(resp); err != nil {
		return "", err
	}
	// Atomisches Schreiben
	tmpFile, err := os.CreateTemp(filepath.Dir(targetPath), ".download-*")
	if err != nil {
		return "", fmt.Errorf("temp-datei erstellen fehlgeschlagen: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if tmpFile != nil {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()
	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		return "", fmt.Errorf("%w: %v", ErrDownloadFailed, err)
	}
	if err := tmpFile.Close(); err != nil {
		return "", fmt.Errorf("datei schliessen fehlgeschlagen: %w", err)
	}
	tmpFile = nil
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return "", fmt.Errorf("datei umbenennen fehlgeschlagen: %w", err)
	}
	return targetPath, nil
}

// GetBaseURL gibt die aktuelle Base-URL zurueck
func (c *Client) GetBaseURL() string { return c.baseURL }

// HasToken prueft ob ein Token konfiguriert ist
func (c *Client) HasToken() bool { return c.token != "" }

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("Accept", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

func (c *Client) handleResponseError(resp *http.Response) error {
	switch resp.StatusCode {
	case http.StatusOK, http.StatusPartialContent:
		return nil
	case http.StatusNotFound:
		return ErrModelNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return ErrUnauthorized
	case http.StatusTooManyRequests:
		return ErrRateLimited
	default:
		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
			return fmt.Errorf("%w: status %d - %s", ErrInvalidResponse, resp.StatusCode, string(body))
		}
		return nil
	}
}

func validateModelID(modelID string) error {
	if modelID == "" {
		return fmt.Errorf("%w: modell-id darf nicht leer sein", ErrInvalidModelID)
	}
	parts := strings.Split(modelID, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("%w: erwartet format 'owner/model'", ErrInvalidModelID)
	}
	return nil
}
