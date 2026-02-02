// cache_test.go - Unit Tests fuer Cache-Management
//
// Autor: Agent 1 - Phase 9
// Datum: 2026-02-01
package huggingface

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGetCacheDir testet die Ermittlung des Cache-Verzeichnisses
func TestGetCacheDir(t *testing.T) {
	// Umgebungsvariablen sichern und zuruecksetzen
	originalHFHubCache := os.Getenv("HF_HUB_CACHE")
	originalHFHome := os.Getenv(EnvHFHome)
	defer func() {
		os.Setenv("HF_HUB_CACHE", originalHFHubCache)
		os.Setenv(EnvHFHome, originalHFHome)
	}()

	tests := []struct {
		name         string
		hfHubCache   string
		hfHome       string
		wantContains string // Teilstring der erwartet wird
	}{
		{
			name:         "HF_HUB_CACHE hat Prioritaet",
			hfHubCache:   "/custom/cache/path",
			hfHome:       "/other/path",
			wantContains: "/custom/cache/path",
		},
		{
			name:         "HF_HOME wird verwendet wenn HF_HUB_CACHE leer",
			hfHubCache:   "",
			hfHome:       "/hf/home",
			wantContains: "hub",
		},
		{
			name:         "Default wird verwendet wenn beide leer",
			hfHubCache:   "",
			hfHome:       "",
			wantContains: "huggingface",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.Setenv("HF_HUB_CACHE", tt.hfHubCache)
			os.Setenv(EnvHFHome, tt.hfHome)

			result := GetCacheDir()

			if tt.hfHubCache != "" && result != tt.hfHubCache {
				t.Errorf("GetCacheDir() = %v, erwartet %v", result, tt.hfHubCache)
			} else if tt.wantContains != "" && !strings.Contains(result, tt.wantContains) {
				t.Errorf("GetCacheDir() = %v, sollte %v enthalten", result, tt.wantContains)
			}
		})
	}
}

// TestModelIDToCacheDir testet die Konvertierung von Model-ID zu Cache-Dir
func TestModelIDToCacheDir(t *testing.T) {
	tests := []struct {
		modelID  string
		expected string
	}{
		{
			modelID:  "google/siglip-base-patch16-224",
			expected: "models--google--siglip-base-patch16-224",
		},
		{
			modelID:  "facebook/dinov2-small",
			expected: "models--facebook--dinov2-small",
		},
		{
			modelID:  "nomic-ai/nomic-embed-vision-v1.5",
			expected: "models--nomic-ai--nomic-embed-vision-v1.5",
		},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			result := modelIDToCacheDir(tt.modelID)
			if result != tt.expected {
				t.Errorf("modelIDToCacheDir(%q) = %q, erwartet %q",
					tt.modelID, result, tt.expected)
			}
		})
	}
}

// TestCacheDirToModelID testet die Rueckkonvertierung
func TestCacheDirToModelID(t *testing.T) {
	tests := []struct {
		cacheDir string
		expected string
	}{
		{
			cacheDir: "models--google--siglip-base-patch16-224",
			expected: "google/siglip-base-patch16-224",
		},
		{
			cacheDir: "models--facebook--dinov2-small",
			expected: "facebook/dinov2-small",
		},
		{
			cacheDir: "models--nomic-ai--nomic-embed-vision-v1.5",
			expected: "nomic-ai/nomic-embed-vision-v1.5",
		},
	}

	for _, tt := range tests {
		t.Run(tt.cacheDir, func(t *testing.T) {
			result := cacheDirToModelID(tt.cacheDir)
			if result != tt.expected {
				t.Errorf("cacheDirToModelID(%q) = %q, erwartet %q",
					tt.cacheDir, result, tt.expected)
			}
		})
	}
}

// TestCacheDirRoundTrip testet Hin- und Rueckkonvertierung
func TestCacheDirRoundTrip(t *testing.T) {
	modelIDs := []string{
		"google/siglip-base-patch16-224",
		"facebook/dinov2-large",
		"nomic-ai/nomic-embed-vision-v1.5",
		"BAAI/EVA02-CLIP-L-14-336",
	}

	for _, modelID := range modelIDs {
		t.Run(modelID, func(t *testing.T) {
			cacheDir := modelIDToCacheDir(modelID)
			result := cacheDirToModelID(cacheDir)
			if result != modelID {
				t.Errorf("RoundTrip fehlgeschlagen: %q -> %q -> %q",
					modelID, cacheDir, result)
			}
		})
	}
}

// TestGetCachedModel testet die Cache-Pruefung
func TestGetCachedModel(t *testing.T) {
	// Temporaeres Test-Verzeichnis erstellen
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	modelID := "test-org/test-model"

	// Zunaechst sollte Modell nicht im Cache sein
	_, found := GetCachedModel(modelID)
	if found {
		t.Error("GetCachedModel sollte false zurueckgeben fuer nicht-existierendes Modell")
	}

	// Cache-Struktur erstellen
	modelDir := modelIDToCacheDir(modelID)
	snapshotPath := filepath.Join(tmpDir, modelDir, CacheSnapshotDir, "main")
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	// Leeres Verzeichnis sollte immer noch nicht gefunden werden
	_, found = GetCachedModel(modelID)
	if found {
		t.Error("GetCachedModel sollte false zurueckgeben fuer leeres Snapshot-Verzeichnis")
	}

	// Datei hinzufuegen
	testFile := filepath.Join(snapshotPath, "config.json")
	if err := os.WriteFile(testFile, []byte("{}"), 0644); err != nil {
		t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
	}

	// Jetzt sollte es gefunden werden
	path, found := GetCachedModel(modelID)
	if !found {
		t.Error("GetCachedModel sollte true zurueckgeben fuer gecachtes Modell")
	}
	if path != snapshotPath {
		t.Errorf("GetCachedModel Pfad = %q, erwartet %q", path, snapshotPath)
	}
}

// TestGetCachedModelWithRevision testet Cache-Pruefung mit Revision
func TestGetCachedModelWithRevision(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	modelID := "test-org/test-model"
	revision := "abc123"

	// Cache-Struktur mit spezifischer Revision erstellen
	modelDir := modelIDToCacheDir(modelID)
	snapshotPath := filepath.Join(tmpDir, modelDir, CacheSnapshotDir, revision)
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	// Datei hinzufuegen
	testFile := filepath.Join(snapshotPath, "model.bin")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
	}

	// "main" sollte nicht gefunden werden
	_, found := GetCachedModel(modelID)
	if found {
		t.Error("GetCachedModel sollte false fuer 'main' zurueckgeben wenn nur andere Revision existiert")
	}

	// Spezifische Revision sollte gefunden werden
	path, found := GetCachedModelWithRevision(modelID, revision)
	if !found {
		t.Error("GetCachedModelWithRevision sollte true zurueckgeben")
	}
	if path != snapshotPath {
		t.Errorf("Pfad = %q, erwartet %q", path, snapshotPath)
	}
}

// TestListCachedModels testet die Auflistung gecachter Modelle
func TestListCachedModels(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	// Leerer Cache
	models, err := ListCachedModels()
	if err != nil {
		t.Fatalf("ListCachedModels fehlgeschlagen: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("Erwartet leere Liste, erhalten %v", models)
	}

	// Modelle hinzufuegen
	testModels := []string{
		"google/siglip-base",
		"facebook/dinov2-small",
	}

	for _, modelID := range testModels {
		modelDir := modelIDToCacheDir(modelID)
		path := filepath.Join(tmpDir, modelDir)
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
		}
	}

	// Jetzt sollten Modelle gefunden werden
	models, err = ListCachedModels()
	if err != nil {
		t.Fatalf("ListCachedModels fehlgeschlagen: %v", err)
	}
	if len(models) != len(testModels) {
		t.Errorf("Erwartet %d Modelle, erhalten %d", len(testModels), len(models))
	}
}

// TestClearModelCache testet das Loeschen eines einzelnen Modells
func TestClearModelCache(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	modelID := "test-org/test-model"

	// Nicht-existierendes Modell loeschen sollte Fehler geben
	err := ClearModelCache(modelID)
	if err != ErrModelNotInCache {
		t.Errorf("Erwartet ErrModelNotInCache, erhalten %v", err)
	}

	// Modell erstellen
	modelDir := modelIDToCacheDir(modelID)
	snapshotPath := filepath.Join(tmpDir, modelDir, CacheSnapshotDir, "main")
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	// Loeschen sollte funktionieren
	if err := ClearModelCache(modelID); err != nil {
		t.Errorf("ClearModelCache fehlgeschlagen: %v", err)
	}

	// Verzeichnis sollte nicht mehr existieren
	if _, err := os.Stat(filepath.Join(tmpDir, modelDir)); !os.IsNotExist(err) {
		t.Error("Model-Verzeichnis sollte geloescht sein")
	}
}

// TestGetCacheSize testet die Groessenberechnung
func TestGetCacheSize(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	// Leerer Cache
	size, err := GetCacheSize()
	if err != nil {
		t.Fatalf("GetCacheSize fehlgeschlagen: %v", err)
	}
	if size != 0 {
		t.Errorf("Erwartet Groesse 0, erhalten %d", size)
	}

	// Datei hinzufuegen
	modelDir := modelIDToCacheDir("test/model")
	filePath := filepath.Join(tmpDir, modelDir, "test.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	testData := make([]byte, 1024) // 1 KB
	if err := os.WriteFile(filePath, testData, 0644); err != nil {
		t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
	}

	// Groesse sollte jetzt 1024 sein
	size, err = GetCacheSize()
	if err != nil {
		t.Fatalf("GetCacheSize fehlgeschlagen: %v", err)
	}
	if size != 1024 {
		t.Errorf("Erwartet Groesse 1024, erhalten %d", size)
	}
}

// TestGetCachedFile testet den Zugriff auf einzelne gecachte Dateien
func TestGetCachedFile(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	modelID := "test/model"
	filename := "config.json"

	// Nicht existierende Datei
	_, found := GetCachedFile(modelID, filename)
	if found {
		t.Error("GetCachedFile sollte false fuer nicht-existierende Datei zurueckgeben")
	}

	// Datei erstellen
	modelDir := modelIDToCacheDir(modelID)
	filePath := filepath.Join(tmpDir, modelDir, CacheSnapshotDir, "main", filename)
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}
	if err := os.WriteFile(filePath, []byte("{}"), 0644); err != nil {
		t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
	}

	// Jetzt sollte sie gefunden werden
	path, found := GetCachedFile(modelID, filename)
	if !found {
		t.Error("GetCachedFile sollte true zurueckgeben")
	}
	if path != filePath {
		t.Errorf("Pfad = %q, erwartet %q", path, filePath)
	}
}

// TestEnsureCacheDir testet das Erstellen des Cache-Verzeichnisses
func TestEnsureCacheDir(t *testing.T) {
	tmpDir := t.TempDir()
	cacheDir := filepath.Join(tmpDir, "new", "cache", "dir")
	os.Setenv("HF_HUB_CACHE", cacheDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	// Verzeichnis existiert noch nicht
	if _, err := os.Stat(cacheDir); !os.IsNotExist(err) {
		t.Fatal("Cache-Verzeichnis sollte noch nicht existieren")
	}

	// EnsureCacheDir aufrufen
	if err := EnsureCacheDir(); err != nil {
		t.Fatalf("EnsureCacheDir fehlgeschlagen: %v", err)
	}

	// Verzeichnis sollte jetzt existieren
	if stat, err := os.Stat(cacheDir); err != nil || !stat.IsDir() {
		t.Error("Cache-Verzeichnis sollte existieren und ein Verzeichnis sein")
	}

	// Erneuter Aufruf sollte nicht fehlschlagen
	if err := EnsureCacheDir(); err != nil {
		t.Errorf("EnsureCacheDir sollte idempotent sein: %v", err)
	}
}

// TestGetCacheInfo testet die Abfrage von Cache-Informationen
func TestGetCacheInfo(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	// Cache-Info fuer leeren Cache
	info, err := GetCacheInfo()
	if err != nil {
		t.Fatalf("GetCacheInfo fehlgeschlagen: %v", err)
	}
	if info.ModelCount != 0 {
		t.Errorf("Erwartet ModelCount 0, erhalten %d", info.ModelCount)
	}

	// Modell mit Dateien erstellen
	modelID := "google/siglip"
	modelDir := modelIDToCacheDir(modelID)
	snapshotPath := filepath.Join(tmpDir, modelDir, CacheSnapshotDir, "main")
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	// Dateien hinzufuegen
	files := []string{"config.json", "model.bin"}
	for _, f := range files {
		path := filepath.Join(snapshotPath, f)
		if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
			t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
		}
	}

	// Erneut Cache-Info abrufen
	info, err = GetCacheInfo()
	if err != nil {
		t.Fatalf("GetCacheInfo fehlgeschlagen: %v", err)
	}
	if info.ModelCount != 1 {
		t.Errorf("Erwartet ModelCount 1, erhalten %d", info.ModelCount)
	}
	if len(info.Models) != 1 {
		t.Fatalf("Erwartet 1 Modell, erhalten %d", len(info.Models))
	}
	if info.Models[0].ModelID != modelID {
		t.Errorf("ModelID = %q, erwartet %q", info.Models[0].ModelID, modelID)
	}
	if len(info.Models[0].Revisions) != 1 || info.Models[0].Revisions[0] != "main" {
		t.Errorf("Revisions = %v, erwartet [main]", info.Models[0].Revisions)
	}
}

// TestClearCache testet das Loeschen des gesamten Caches
func TestClearCache(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	// Mehrere Modelle erstellen
	models := []string{"org1/model1", "org2/model2"}
	for _, modelID := range models {
		modelDir := modelIDToCacheDir(modelID)
		path := filepath.Join(tmpDir, modelDir, "test.txt")
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
		}
		if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
			t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
		}
	}

	// Pruefen dass Modelle existieren
	list, _ := ListCachedModels()
	if len(list) != 2 {
		t.Fatalf("Erwartet 2 Modelle vor ClearCache, erhalten %d", len(list))
	}

	// Cache loeschen
	if err := ClearCache(); err != nil {
		t.Fatalf("ClearCache fehlgeschlagen: %v", err)
	}

	// Pruefen dass keine Modelle mehr existieren
	list, _ = ListCachedModels()
	if len(list) != 0 {
		t.Errorf("Erwartet 0 Modelle nach ClearCache, erhalten %d", len(list))
	}
}

// TestGetModelCacheSize testet die Groessenberechnung fuer ein einzelnes Modell
func TestGetModelCacheSize(t *testing.T) {
	tmpDir := t.TempDir()
	os.Setenv("HF_HUB_CACHE", tmpDir)
	defer os.Unsetenv("HF_HUB_CACHE")

	modelID := "test/model"

	// Nicht existierendes Modell
	_, err := GetModelCacheSize(modelID)
	if err != ErrModelNotInCache {
		t.Errorf("Erwartet ErrModelNotInCache, erhalten %v", err)
	}

	// Modell erstellen
	modelDir := modelIDToCacheDir(modelID)
	filePath := filepath.Join(tmpDir, modelDir, "data.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		t.Fatalf("Verzeichnis erstellen fehlgeschlagen: %v", err)
	}

	testData := make([]byte, 2048) // 2 KB
	if err := os.WriteFile(filePath, testData, 0644); err != nil {
		t.Fatalf("Testdatei erstellen fehlgeschlagen: %v", err)
	}

	// Groesse abrufen
	size, err := GetModelCacheSize(modelID)
	if err != nil {
		t.Fatalf("GetModelCacheSize fehlgeschlagen: %v", err)
	}
	if size != 2048 {
		t.Errorf("Erwartet Groesse 2048, erhalten %d", size)
	}
}
