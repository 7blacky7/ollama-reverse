// cache.go - Cache-Management fuer HuggingFace Modelle
// Kompatibel mit Python huggingface_hub Cache-Struktur.
// Autor: Agent 1 - Phase 9
package huggingface

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Cache-Konstanten
const (
	DefaultCacheSubdir = "huggingface/hub"
	CacheRefDir        = "refs"
	CacheBlobDir       = "blobs"
	CacheSnapshotDir   = "snapshots"
	CacheModelPrefix   = "models--"
)

// Cache-Fehler
var (
	ErrCacheNotFound     = errors.New("cache-verzeichnis nicht gefunden")
	ErrModelNotInCache   = errors.New("modell nicht im cache")
	ErrCacheCorrupted    = errors.New("cache-struktur beschaedigt")
	ErrCacheAccessDenied = errors.New("zugriff auf cache verweigert")
)

// CachedModel repraesentiert ein gecachtes Modell
type CachedModel struct {
	ModelID   string
	CacheDir  string
	Revisions []string
	TotalSize int64
	FileCount int
}

// CacheInfo enthaelt Informationen ueber den gesamten Cache
type CacheInfo struct {
	CacheDir   string
	TotalSize  int64
	ModelCount int
	Models     []CachedModel
}

// GetCacheDir gibt das Cache-Verzeichnis zurueck
func GetCacheDir() string {
	if cacheDir := os.Getenv("HF_HUB_CACHE"); cacheDir != "" {
		return cacheDir
	}
	if hfHome := os.Getenv(EnvHFHome); hfHome != "" {
		return filepath.Join(hfHome, "hub")
	}
	return getDefaultCacheDir()
}

func getDefaultCacheDir() string {
	var baseDir string
	switch runtime.GOOS {
	case "windows":
		if userProfile := os.Getenv("USERPROFILE"); userProfile != "" {
			baseDir = filepath.Join(userProfile, ".cache")
		} else {
			baseDir = filepath.Join(os.TempDir(), "huggingface_cache")
		}
	default:
		if xdgCache := os.Getenv("XDG_CACHE_HOME"); xdgCache != "" {
			baseDir = xdgCache
		} else if home, err := os.UserHomeDir(); err == nil {
			baseDir = filepath.Join(home, ".cache")
		} else {
			baseDir = filepath.Join(os.TempDir(), "huggingface_cache")
		}
	}
	return filepath.Join(baseDir, DefaultCacheSubdir)
}

// GetCachedModel prueft ob ein Modell im Cache ist
func GetCachedModel(modelID string) (string, bool) {
	return GetCachedModelWithRevision(modelID, "main")
}

// GetCachedModelWithRevision prueft den Cache fuer eine spezifische Revision
func GetCachedModelWithRevision(modelID, revision string) (string, bool) {
	cacheDir := GetCacheDir()
	snapshotPath := filepath.Join(cacheDir, modelIDToCacheDir(modelID), CacheSnapshotDir, revision)
	if stat, err := os.Stat(snapshotPath); err == nil && stat.IsDir() {
		if entries, err := os.ReadDir(snapshotPath); err == nil && len(entries) > 0 {
			return snapshotPath, true
		}
	}
	return "", false
}

// GetCachedFile gibt den Pfad zu einer spezifischen Datei im Cache zurueck
func GetCachedFile(modelID, filename string) (string, bool) {
	return GetCachedFileWithRevision(modelID, filename, "main")
}

// GetCachedFileWithRevision gibt den Pfad zu einer Datei mit Revision zurueck
func GetCachedFileWithRevision(modelID, filename, revision string) (string, bool) {
	filePath := filepath.Join(GetCacheDir(), modelIDToCacheDir(modelID), CacheSnapshotDir, revision, filename)
	if _, err := os.Stat(filePath); err == nil {
		return filePath, true
	}
	return "", false
}

// ClearCache loescht den gesamten HuggingFace Cache
func ClearCache() error {
	cacheDir := GetCacheDir()
	if _, err := os.Stat(cacheDir); os.IsNotExist(err) {
		return nil
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		if os.IsPermission(err) {
			return ErrCacheAccessDenied
		}
		return fmt.Errorf("cache lesen fehlgeschlagen: %w", err)
	}
	var lastErr error
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), CacheModelPrefix) {
			if err := os.RemoveAll(filepath.Join(cacheDir, entry.Name())); err != nil {
				lastErr = err
			}
		}
	}
	return lastErr
}

// ClearModelCache loescht den Cache fuer ein spezifisches Modell
func ClearModelCache(modelID string) error {
	modelPath := filepath.Join(GetCacheDir(), modelIDToCacheDir(modelID))
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return ErrModelNotInCache
	}
	return os.RemoveAll(modelPath)
}

// EnsureCacheDir stellt sicher, dass das Cache-Verzeichnis existiert
func EnsureCacheDir() error {
	return os.MkdirAll(GetCacheDir(), 0755)
}

// GetCacheSize berechnet die Gesamtgroesse des Caches in Bytes
func GetCacheSize() (int64, error) {
	return getDirSize(GetCacheDir())
}

// GetModelCacheSize berechnet die Groesse eines spezifischen Modells im Cache
func GetModelCacheSize(modelID string) (int64, error) {
	modelPath := filepath.Join(GetCacheDir(), modelIDToCacheDir(modelID))
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return 0, ErrModelNotInCache
	}
	return getDirSize(modelPath)
}

// GetCacheInfo gibt detaillierte Informationen ueber den Cache zurueck
func GetCacheInfo() (*CacheInfo, error) {
	cacheDir := GetCacheDir()
	if _, err := os.Stat(cacheDir); os.IsNotExist(err) {
		return &CacheInfo{CacheDir: cacheDir}, nil
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return nil, fmt.Errorf("cache lesen fehlgeschlagen: %w", err)
	}
	info := &CacheInfo{CacheDir: cacheDir, Models: make([]CachedModel, 0)}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), CacheModelPrefix) {
			continue
		}
		modelPath := filepath.Join(cacheDir, entry.Name())
		cachedModel := CachedModel{ModelID: cacheDirToModelID(entry.Name()), CacheDir: modelPath}
		snapshotPath := filepath.Join(modelPath, CacheSnapshotDir)
		if revisions, err := os.ReadDir(snapshotPath); err == nil {
			for _, rev := range revisions {
				if rev.IsDir() {
					cachedModel.Revisions = append(cachedModel.Revisions, rev.Name())
				}
			}
		}
		cachedModel.TotalSize, cachedModel.FileCount = getDirSizeAndCount(modelPath)
		info.Models = append(info.Models, cachedModel)
		info.TotalSize += cachedModel.TotalSize
		info.ModelCount++
	}
	return info, nil
}

// ListCachedModels gibt eine Liste aller gecachten Modelle zurueck
func ListCachedModels() ([]string, error) {
	cacheDir := GetCacheDir()
	if _, err := os.Stat(cacheDir); os.IsNotExist(err) {
		return []string{}, nil
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return nil, fmt.Errorf("cache lesen fehlgeschlagen: %w", err)
	}
	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), CacheModelPrefix) {
			models = append(models, cacheDirToModelID(entry.Name()))
		}
	}
	return models, nil
}

func modelIDToCacheDir(modelID string) string {
	return CacheModelPrefix + strings.ReplaceAll(modelID, "/", "--")
}

func cacheDirToModelID(cacheDir string) string {
	return strings.Replace(strings.TrimPrefix(cacheDir, CacheModelPrefix), "--", "/", 1)
}

func getDirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}

func getDirSizeAndCount(path string) (int64, int) {
	var size int64
	var count int
	filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			size += info.Size()
			count++
		}
		return nil
	})
	return size, count
}
