// Package vision - HuggingFace Integration fuer Vision Registry.
//
// MODUL: registry_hf
// ZWECK: Ermoeglicht das automatische Laden von Vision-Modellen von HuggingFace
// INPUT: HuggingFace Model-ID (z.B. "google/siglip-base-patch16-224"), HFOptions
// OUTPUT: VisionEncoder mit geladenem Modell
// NEBENEFFEKTE: Download von Modellen, Dateisystem-Zugriffe, Netzwerk-Operationen
// ABHAENGIGKEITEN: registry.go, factory.go, options.go, net/http, os, path/filepath
// HINWEISE: Unterstuetzt Caching, Revisions und Progress-Callbacks

package vision

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ============================================================================
// Konstanten fuer HuggingFace
// ============================================================================

const (
	// HF API und Download URLs
	hfAPIBaseURL      = "https://huggingface.co/api/models/"
	hfDownloadBaseURL = "https://huggingface.co/"
	hfResolveEndpoint = "/resolve/"

	// Default-Werte
	defaultCacheDir     = ".cache/huggingface/hub"
	defaultRevision     = "main"
	httpTimeoutSeconds  = 300
	downloadBufferSize  = 32 * 1024   // 32 KB Buffer
	progressUpdateBytes = 1024 * 1024 // Progress alle 1MB updaten
)

// ============================================================================
// Fehler-Definitionen fuer HuggingFace
// ============================================================================

var (
	// ErrInvalidHFModelID wird zurueckgegeben wenn die Model-ID ungueltig ist.
	ErrInvalidHFModelID = errors.New("vision/hf: invalid HuggingFace model ID")

	// ErrModelNotOnHF wird zurueckgegeben wenn das Modell nicht auf HF existiert.
	ErrModelNotOnHF = errors.New("vision/hf: model not found on HuggingFace")

	// ErrNoGGUFFound wird zurueckgegeben wenn keine GGUF-Datei im Repo ist.
	ErrNoGGUFFound = errors.New("vision/hf: no GGUF file found in repository")

	// ErrDownloadFailed wird zurueckgegeben wenn der Download fehlschlaegt.
	ErrDownloadFailed = errors.New("vision/hf: download failed")

	// ErrUnsupportedModelType wird zurueckgegeben wenn der Modell-Typ nicht unterstuetzt wird.
	ErrUnsupportedModelType = errors.New("vision/hf: unsupported model type")
)

// HFError repraesentiert einen HuggingFace-spezifischen Fehler.
type HFError struct {
	Op      string // Operation (z.B. "download", "detect")
	ModelID string // HuggingFace Model-ID
	Err     error  // Urspruenglicher Fehler
}

// Error implementiert das error Interface.
func (e *HFError) Error() string {
	return fmt.Sprintf("vision/hf: %s '%s': %v", e.Op, e.ModelID, e.Err)
}

// Unwrap gibt den urspruenglichen Fehler zurueck.
func (e *HFError) Unwrap() error {
	return e.Err
}

// ============================================================================
// HFOptions - Konfiguration fuer HuggingFace Downloads
// ============================================================================

// HFOptions enthaelt Optionen fuer HuggingFace Model-Downloads.
type HFOptions struct {
	Revision    string                           // Git-Revision (Branch, Tag, Commit)
	ForceReload bool                             // Cache ignorieren und neu laden
	CacheDir    string                           // Cache-Verzeichnis
	OnProgress  func(status string, pct float64) // Progress-Callback
	Token       string                           // HF Auth-Token (optional)
	LoadOptions LoadOptions                      // Vision Encoder LoadOptions
}

// HFOption ist eine funktionale Option fuer HFOptions.
type HFOption func(*HFOptions)

// ============================================================================
// DefaultHFOptions - Standard-Konfiguration
// ============================================================================

// DefaultHFOptions gibt eine Standard-Konfiguration fuer HF-Downloads zurueck.
func DefaultHFOptions() HFOptions {
	homeDir, _ := os.UserHomeDir()
	return HFOptions{
		Revision:    defaultRevision,
		ForceReload: false,
		CacheDir:    filepath.Join(homeDir, defaultCacheDir),
		OnProgress:  nil,
		Token:       os.Getenv("HF_TOKEN"),
		LoadOptions: DefaultLoadOptions(),
	}
}

// ============================================================================
// Functional Options - Builder-Funktionen fuer HF
// ============================================================================

// WithRevision setzt die Git-Revision (Branch, Tag, Commit-Hash).
func WithHFRevision(rev string) HFOption {
	return func(o *HFOptions) {
		if rev != "" {
			o.Revision = rev
		}
	}
}

// WithForceReload erzwingt einen Neudownload auch wenn gecached.
func WithForceReload() HFOption {
	return func(o *HFOptions) {
		o.ForceReload = true
	}
}

// WithHFCacheDir setzt das Cache-Verzeichnis.
func WithHFCacheDir(dir string) HFOption {
	return func(o *HFOptions) {
		if dir != "" {
			o.CacheDir = dir
		}
	}
}

// WithProgress setzt den Progress-Callback.
func WithProgress(fn func(status string, pct float64)) HFOption {
	return func(o *HFOptions) {
		o.OnProgress = fn
	}
}

// WithHFToken setzt den HuggingFace Auth-Token.
func WithHFToken(token string) HFOption {
	return func(o *HFOptions) {
		o.Token = token
	}
}

// WithHFLoadOptions setzt die LoadOptions fuer den Encoder.
func WithHFLoadOptions(opts LoadOptions) HFOption {
	return func(o *HFOptions) {
		o.LoadOptions = opts
	}
}

// ============================================================================
// IsHuggingFaceID - Prueft ob String eine HF Model-ID ist
// ============================================================================

// hfModelIDRegex validiert HuggingFace Model-IDs.
// Format: "org/model" oder "org/model:revision"
var hfModelIDRegex = regexp.MustCompile(`^[\w\-\.]+/[\w\-\.]+(?::[\w\-\.]+)?$`)

// IsHuggingFaceID prueft ob ein String eine gueltige HuggingFace Model-ID ist.
// Gueltige Formate:
//   - "google/siglip-base-patch16-224"
//   - "openai/clip-vit-base-patch32:v1.0"
func IsHuggingFaceID(id string) bool {
	if id == "" {
		return false
	}
	// Lokale Pfade ausschliessen
	if strings.ContainsAny(id, `\/`) && (strings.HasPrefix(id, "/") ||
		strings.HasPrefix(id, "./") || strings.HasPrefix(id, "..") ||
		len(id) > 2 && id[1] == ':') {
		return false
	}
	return hfModelIDRegex.MatchString(id)
}

// ============================================================================
// ParseHuggingFaceID - Model-ID parsen
// ============================================================================

// HFModelRef repraesentiert eine geparste HuggingFace Model-Referenz.
type HFModelRef struct {
	Org      string // Organisation/User
	Model    string // Modell-Name
	Revision string // Revision (leer = main)
}

// ParseHuggingFaceID parst eine HuggingFace Model-ID.
func ParseHuggingFaceID(id string) (HFModelRef, error) {
	if !IsHuggingFaceID(id) {
		return HFModelRef{}, ErrInvalidHFModelID
	}

	// Revision abtrennen wenn vorhanden
	ref := HFModelRef{Revision: defaultRevision}
	colonIdx := strings.LastIndex(id, ":")
	if colonIdx > 0 && !strings.Contains(id[colonIdx:], "/") {
		ref.Revision = id[colonIdx+1:]
		id = id[:colonIdx]
	}

	// Org und Model trennen
	parts := strings.SplitN(id, "/", 2)
	if len(parts) != 2 {
		return HFModelRef{}, ErrInvalidHFModelID
	}

	ref.Org = parts[0]
	ref.Model = parts[1]
	return ref, nil
}

// FullID gibt die vollstaendige Model-ID zurueck.
func (r HFModelRef) FullID() string {
	return r.Org + "/" + r.Model
}

// ============================================================================
// LoadFromHuggingFace - Hauptfunktion
// ============================================================================

// LoadFromHuggingFace laedt ein Vision-Modell von HuggingFace.
// Unterstuetzte Modell-Typen werden automatisch erkannt.
func LoadFromHuggingFace(modelID string, opts ...HFOption) (VisionEncoder, error) {
	// Optionen anwenden
	hfOpts := DefaultHFOptions()
	for _, opt := range opts {
		opt(&hfOpts)
	}

	// Model-ID parsen
	ref, err := ParseHuggingFaceID(modelID)
	if err != nil {
		return nil, err
	}

	// Revision aus Options uebernehmen wenn nicht in ID
	if ref.Revision == defaultRevision && hfOpts.Revision != defaultRevision {
		ref.Revision = hfOpts.Revision
	}

	// Progress-Callback
	reportProgress := func(status string, pct float64) {
		if hfOpts.OnProgress != nil {
			hfOpts.OnProgress(status, pct)
		}
	}

	// Cache-Pfad ermitteln
	cachedPath := getCachePath(hfOpts.CacheDir, ref)
	reportProgress("Pruefe Cache...", 0)

	// Pruefen ob bereits gecached
	if !hfOpts.ForceReload && fileExists(cachedPath) {
		reportProgress("Lade aus Cache...", 50)
		return loadCachedModel(cachedPath, hfOpts.LoadOptions)
	}

	// GGUF-Datei im Repository finden
	reportProgress("Suche GGUF-Datei...", 10)
	ggufFile, err := findGGUFInRepo(ref, hfOpts.Token)
	if err != nil {
		return nil, &HFError{Op: "find_gguf", ModelID: modelID, Err: err}
	}

	// Download mit Progress
	reportProgress("Starte Download...", 15)
	err = downloadGGUF(ref, ggufFile, cachedPath, hfOpts)
	if err != nil {
		return nil, &HFError{Op: "download", ModelID: modelID, Err: err}
	}

	// Modell laden
	reportProgress("Lade Modell...", 95)
	return loadCachedModel(cachedPath, hfOpts.LoadOptions)
}

// ============================================================================
// Cache-Management Funktionen
// ============================================================================

// getCachePath ermittelt den Cache-Pfad fuer ein Modell.
func getCachePath(cacheDir string, ref HFModelRef) string {
	// Format: cacheDir/models--org--model/snapshots/revision/model.gguf
	safeOrg := strings.ReplaceAll(ref.Org, "/", "--")
	safeModel := strings.ReplaceAll(ref.Model, "/", "--")
	modelDir := fmt.Sprintf("models--%s--%s", safeOrg, safeModel)
	return filepath.Join(cacheDir, modelDir, "snapshots", ref.Revision, "model.gguf")
}

// fileExists prueft ob eine Datei existiert.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// loadCachedModel laedt ein gecachtes Modell.
func loadCachedModel(path string, opts LoadOptions) (VisionEncoder, error) {
	// Modell-Typ automatisch erkennen
	modelType, err := AutoDetectEncoder(path)
	if err != nil {
		return nil, &HFError{Op: "detect_type", ModelID: path, Err: err}
	}

	// Encoder ueber DefaultRegistry erstellen
	return DefaultRegistry.Create(modelType, path, opts)
}

// ============================================================================
// HuggingFace API Funktionen
// ============================================================================

// hfRepoFile repraesentiert eine Datei im HF Repository.
type hfRepoFile struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size"`
}

// hfRepoInfo repraesentiert Repository-Informationen.
type hfRepoInfo struct {
	Siblings []hfRepoFile `json:"siblings"`
}

// findGGUFInRepo findet die GGUF-Datei in einem HF Repository.
func findGGUFInRepo(ref HFModelRef, token string) (string, error) {
	// API-URL bauen
	apiURL := hfAPIBaseURL + ref.FullID() + "?revision=" + ref.Revision

	// HTTP Request erstellen
	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil {
		return "", err
	}

	// Auth-Header wenn Token vorhanden
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	// Request ausfuehren
	client := &http.Client{Timeout: time.Duration(httpTimeoutSeconds) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Status pruefen
	if resp.StatusCode == http.StatusNotFound {
		return "", ErrModelNotOnHF
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HF API returned status %d", resp.StatusCode)
	}

	// Response parsen
	var info hfRepoInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", err
	}

	// GGUF-Datei suchen
	for _, file := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(file.Filename), ".gguf") {
			return file.Filename, nil
		}
	}

	return "", ErrNoGGUFFound
}

// ============================================================================
// Download Funktionen
// ============================================================================

// downloadGGUF laedt eine GGUF-Datei von HuggingFace.
func downloadGGUF(ref HFModelRef, filename string, destPath string, opts HFOptions) error {
	// Download-URL bauen
	downloadURL := hfDownloadBaseURL + ref.FullID() + hfResolveEndpoint + ref.Revision + "/" + filename

	// HTTP Request erstellen
	req, err := http.NewRequest(http.MethodGet, downloadURL, nil)
	if err != nil {
		return err
	}

	// Auth-Header wenn Token vorhanden
	if opts.Token != "" {
		req.Header.Set("Authorization", "Bearer "+opts.Token)
	}

	// Request ausfuehren
	client := &http.Client{Timeout: time.Duration(httpTimeoutSeconds) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Status pruefen
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: status %d", ErrDownloadFailed, resp.StatusCode)
	}

	// Zielverzeichnis erstellen
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	// Temp-Datei fuer atomischen Schreibvorgang
	tempPath := destPath + ".tmp"
	destFile, err := os.Create(tempPath)
	if err != nil {
		return err
	}
	defer func() {
		destFile.Close()
		os.Remove(tempPath) // Cleanup bei Fehler
	}()

	// Download mit Progress
	totalSize := resp.ContentLength
	downloaded := int64(0)
	lastUpdate := int64(0)
	buf := make([]byte, downloadBufferSize)

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := destFile.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)

			// Progress updaten
			if downloaded-lastUpdate >= progressUpdateBytes && opts.OnProgress != nil {
				pct := float64(downloaded) / float64(totalSize) * 80 // 15-95%
				opts.OnProgress("Download...", 15+pct)
				lastUpdate = downloaded
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	// Datei schliessen und umbenennen
	if err := destFile.Close(); err != nil {
		return err
	}

	return os.Rename(tempPath, destPath)
}

// ============================================================================
// Convenience-Funktionen
// ============================================================================

// MustLoadFromHuggingFace laedt ein Modell und panict bei Fehler.
// Nuetzlich fuer init()-Funktionen.
func MustLoadFromHuggingFace(modelID string, opts ...HFOption) VisionEncoder {
	encoder, err := LoadFromHuggingFace(modelID, opts...)
	if err != nil {
		panic(fmt.Sprintf("vision/hf: failed to load '%s': %v", modelID, err))
	}
	return encoder
}

// ClearCache loescht den HuggingFace Cache fuer ein bestimmtes Modell.
func ClearCache(modelID string, cacheDir string) error {
	ref, err := ParseHuggingFaceID(modelID)
	if err != nil {
		return err
	}

	if cacheDir == "" {
		homeDir, _ := os.UserHomeDir()
		cacheDir = filepath.Join(homeDir, defaultCacheDir)
	}

	// Model-Verzeichnis ermitteln
	safeOrg := strings.ReplaceAll(ref.Org, "/", "--")
	safeModel := strings.ReplaceAll(ref.Model, "/", "--")
	modelDir := filepath.Join(cacheDir, fmt.Sprintf("models--%s--%s", safeOrg, safeModel))

	return os.RemoveAll(modelDir)
}

// IsCached prueft ob ein Modell bereits gecached ist.
func IsCached(modelID string, opts ...HFOption) bool {
	hfOpts := DefaultHFOptions()
	for _, opt := range opts {
		opt(&hfOpts)
	}

	ref, err := ParseHuggingFaceID(modelID)
	if err != nil {
		return false
	}

	if ref.Revision == defaultRevision && hfOpts.Revision != defaultRevision {
		ref.Revision = hfOpts.Revision
	}

	cachedPath := getCachePath(hfOpts.CacheDir, ref)
	return fileExists(cachedPath)
}
