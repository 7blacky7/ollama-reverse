// download.go - Download-Logik fuer HuggingFace Modelle mit Progress-Callback
// Unterstuetzt Progress-Callbacks, Revisions und parallele Downloads.
// Autor: Agent 1 - Phase 9
package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Download-Konstanten
const (
	DefaultChunkSize       = 1024 * 1024         // 1 MB
	MaxDownloadRetries     = 3
	DownloadRetryDelay     = 2 * time.Second
	ProgressUpdateInterval = 100 * time.Millisecond
	DefaultParallelism     = 4
)

// ModelDownloadResult enthaelt das Ergebnis eines Model-Downloads
type ModelDownloadResult struct {
	ModelID      string
	Revision     string
	CachePath    string
	Files        []DownloadedFile
	TotalSize    int64
	DownloadTime time.Duration
}

// DownloadedFile repraesentiert eine heruntergeladene Datei
type DownloadedFile struct {
	Filename  string
	LocalPath string
	Size      int64
	FromCache bool
}

// ProgressCallback wird waehrend des Downloads aufgerufen
type ProgressCallback func(downloaded, total int64)

// DownloadOption konfiguriert einen Download
type DownloadOption func(*downloadConfig)

type downloadConfig struct {
	revision        string
	files           []string
	progressFn      ProgressCallback
	parallelism     int
	includePatterns []string
	excludePatterns []string
	resume          bool
}

// WithDownloadRevision setzt die Git-Revision fuer den Download
func WithDownloadRevision(revision string) DownloadOption {
	return func(cfg *downloadConfig) { cfg.revision = revision }
}

// WithDownloadFiles begrenzt den Download auf spezifische Dateien
func WithDownloadFiles(files ...string) DownloadOption {
	return func(cfg *downloadConfig) { cfg.files = files }
}

// WithDownloadProgress setzt den Progress-Callback
func WithDownloadProgress(fn ProgressCallback) DownloadOption {
	return func(cfg *downloadConfig) { cfg.progressFn = fn }
}

// WithDownloadParallelism setzt die Anzahl paralleler Downloads
func WithDownloadParallelism(n int) DownloadOption {
	return func(cfg *downloadConfig) {
		if n > 0 {
			cfg.parallelism = n
		}
	}
}

// WithIncludePatterns filtert Dateien nach Glob-Patterns
func WithIncludePatterns(patterns ...string) DownloadOption {
	return func(cfg *downloadConfig) { cfg.includePatterns = patterns }
}

// WithExcludePatterns schliesst Dateien nach Glob-Patterns aus
func WithExcludePatterns(patterns ...string) DownloadOption {
	return func(cfg *downloadConfig) { cfg.excludePatterns = patterns }
}

// WithResume aktiviert das Fortsetzen abgebrochener Downloads
func WithResume(resume bool) DownloadOption {
	return func(cfg *downloadConfig) { cfg.resume = resume }
}

// DownloadModel laedt ein komplettes Modell herunter
func DownloadModel(modelID string, opts ...DownloadOption) (*ModelDownloadResult, error) {
	client := NewClient()
	return client.DownloadModelWithContext(context.Background(), modelID, opts...)
}

// DownloadModelWithContext laedt ein Modell mit Context herunter
func (c *Client) DownloadModelWithContext(ctx context.Context, modelID string, opts ...DownloadOption) (*ModelDownloadResult, error) {
	startTime := time.Now()
	cfg := &downloadConfig{revision: "main", parallelism: DefaultParallelism, resume: true}
	for _, opt := range opts {
		opt(cfg)
	}
	info, err := c.GetModelInfoWithContext(ctx, modelID)
	if err != nil {
		return nil, fmt.Errorf("model-info abrufen fehlgeschlagen: %w", err)
	}
	filesToDownload := filterDownloadFiles(info.Siblings, cfg)
	if len(filesToDownload) == 0 {
		return nil, fmt.Errorf("keine dateien zum download gefunden")
	}
	var totalSize int64
	for _, f := range filesToDownload {
		totalSize += f.Size
	}
	cacheDir := GetCacheDir()
	modelCacheDir := filepath.Join(cacheDir, "models--"+strings.ReplaceAll(modelID, "/", "--"))
	snapshotDir := filepath.Join(modelCacheDir, "snapshots", cfg.revision)

	var downloadedBytes int64
	var progressMu sync.Mutex
	lastProgressUpdate := time.Now()
	updateProgress := func(bytes int64) {
		if cfg.progressFn == nil {
			return
		}
		progressMu.Lock()
		defer progressMu.Unlock()
		downloadedBytes += bytes
		now := time.Now()
		if now.Sub(lastProgressUpdate) >= ProgressUpdateInterval {
			cfg.progressFn(downloadedBytes, totalSize)
			lastProgressUpdate = now
		}
	}

	results := make([]DownloadedFile, 0, len(filesToDownload))
	var resultsMu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(filesToDownload))
	semaphore := make(chan struct{}, cfg.parallelism)

	for _, file := range filesToDownload {
		wg.Add(1)
		go func(f APISibling) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			localPath := filepath.Join(snapshotDir, f.Filename)
			fromCache := false
			if stat, err := os.Stat(localPath); err == nil && stat.Size() == f.Size {
				fromCache = true
				updateProgress(f.Size)
			} else {
				if err := c.downloadFileWithProgress(ctx, modelID, f.Filename, cfg.revision, localPath, updateProgress); err != nil {
					errChan <- fmt.Errorf("download von %s fehlgeschlagen: %w", f.Filename, err)
					return
				}
			}
			resultsMu.Lock()
			results = append(results, DownloadedFile{Filename: f.Filename, LocalPath: localPath, Size: f.Size, FromCache: fromCache})
			resultsMu.Unlock()
		}(file)
	}
	wg.Wait()
	close(errChan)

	var downloadErrors []error
	for err := range errChan {
		downloadErrors = append(downloadErrors, err)
	}
	if len(downloadErrors) > 0 {
		return nil, fmt.Errorf("download-fehler: %v", downloadErrors)
	}
	if cfg.progressFn != nil {
		cfg.progressFn(totalSize, totalSize)
	}
	return &ModelDownloadResult{
		ModelID: modelID, Revision: cfg.revision, CachePath: snapshotDir,
		Files: results, TotalSize: totalSize, DownloadTime: time.Since(startTime),
	}, nil
}

// DownloadWithProgress laedt ein Modell mit Progress-Callback herunter
func DownloadWithProgress(modelID string, progress func(downloaded, total int64)) error {
	_, err := DownloadModel(modelID, WithDownloadProgress(progress))
	return err
}

func (c *Client) downloadFileWithProgress(ctx context.Context, modelID, filename, revision, targetPath string, progressFn func(int64)) error {
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return fmt.Errorf("verzeichnis erstellen fehlgeschlagen: %w", err)
	}
	url := fmt.Sprintf("%s/%s/resolve/%s/%s", c.baseURL, modelID, revision, filename)
	var lastErr error
	for attempt := 0; attempt < MaxDownloadRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(DownloadRetryDelay):
			}
		}
		if err := c.doDownload(ctx, url, targetPath, progressFn); err != nil {
			lastErr = err
			continue
		}
		return nil
	}
	return fmt.Errorf("download nach %d versuchen fehlgeschlagen: %w", MaxDownloadRetries, lastErr)
}

func (c *Client) doDownload(ctx context.Context, url, targetPath string, progressFn func(int64)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	c.setHeaders(req)
	var existingSize int64
	tmpPath := targetPath + ".download"
	if stat, err := os.Stat(tmpPath); err == nil {
		existingSize = stat.Size()
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK && existingSize > 0 {
		existingSize = 0
		os.Remove(tmpPath)
	} else if err := c.handleResponseError(resp); err != nil {
		return err
	}
	flags := os.O_WRONLY | os.O_CREATE
	if existingSize > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(tmpPath, flags, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	buf := make([]byte, DefaultChunkSize)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := file.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			if progressFn != nil {
				progressFn(int64(n))
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, targetPath)
}

func filterDownloadFiles(siblings []APISibling, cfg *downloadConfig) []APISibling {
	if len(cfg.files) > 0 {
		fileSet := make(map[string]bool)
		for _, f := range cfg.files {
			fileSet[f] = true
		}
		var result []APISibling
		for _, s := range siblings {
			if fileSet[s.Filename] {
				result = append(result, s)
			}
		}
		return result
	}
	var result []APISibling
	for _, s := range siblings {
		if len(cfg.includePatterns) > 0 {
			matched := false
			for _, pattern := range cfg.includePatterns {
				if m, _ := filepath.Match(pattern, s.Filename); m {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		excluded := false
		for _, pattern := range cfg.excludePatterns {
			if m, _ := filepath.Match(pattern, s.Filename); m {
				excluded = true
				break
			}
		}
		if excluded {
			continue
		}
		result = append(result, s)
	}
	return result
}
