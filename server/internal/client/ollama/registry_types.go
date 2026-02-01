// Package ollama - Registry Types und Konfiguration
//
// Diese Datei enthält:
// - Fehler-Definitionen (ErrModelNotFound, ErrManifestInvalid, etc.)
// - Konstanten (DefaultChunkingThreshold, DefaultMask)
// - Error-Typ mit JSON-Marshaling
// - Registry-Struct und Konfiguration
// - Cache-Funktionen
package ollama

import (
	"cmp"
	"crypto"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/internal/names"
)

// Errors - Zentrale Fehlerdefinitionen
var (
	// ErrModelNotFound wird zurückgegeben wenn ein Manifest nicht gefunden wird
	ErrModelNotFound = errors.New("model not found")

	// ErrManifestInvalid wird zurückgegeben bei ungültigem Manifest
	ErrManifestInvalid = errors.New("invalid manifest")

	// ErrNameInvalid wird zurückgegeben bei ungültigem Namen
	ErrNameInvalid = errors.New("invalid or missing name")

	// ErrCached wird an Trace.PushUpdate übergeben wenn Layer bereits existiert
	ErrCached = errors.New("cached")

	// ErrIncomplete wird zurückgegeben bei unvollständigem Pull
	ErrIncomplete = errors.New("incomplete")
)

// Defaults - Standard-Konfigurationswerte
const (
	// DefaultChunkingThreshold ist der Schwellwert für Chunk-Downloads (64 MB)
	DefaultChunkingThreshold = 64 << 20
)

// defaultCache - Singleton für den Standard-Cache
var defaultCache = sync.OnceValues(func() (*blob.DiskCache, error) {
	dir := os.Getenv("OLLAMA_MODELS")
	if dir == "" {
		home, _ := os.UserHomeDir()
		home = cmp.Or(home, ".")
		dir = filepath.Join(home, ".ollama", "models")
	}
	return blob.Open(dir)
})

// DefaultCache gibt den Standard-Cache zurück, konfiguriert aus OLLAMA_MODELS
// oder $HOME/.ollama/models
func DefaultCache() (*blob.DiskCache, error) {
	return defaultCache()
}

// Error ist der Standard-Fehlertyp der Ollama API
type Error struct {
	status  int    `json:"-"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Temporary prüft ob der Fehler temporär ist (5xx Status)
func (e *Error) Temporary() bool {
	return e.status >= 500
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("registry responded with status ")
	b.WriteString(strconv.Itoa(e.status))
	if e.Code != "" {
		b.WriteString(": code ")
		b.WriteString(e.Code)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

func (e *Error) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Int("status", e.status),
		slog.String("code", e.Code),
		slog.String("message", e.Message),
	)
}

// UnmarshalJSON implementiert json.Unmarshaler
func (e *Error) UnmarshalJSON(b []byte) error {
	type E Error
	var v struct {
		Code   string
		Error  string
		Errors []E
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	if v.Error != "" {
		e.Code = v.Code
		e.Message = v.Error
		return nil
	}
	if len(v.Errors) == 0 {
		return fmt.Errorf("no messages in error response: %s", string(b))
	}
	*e = Error(v.Errors[0])
	return nil
}

// DefaultMask - Standard-Maske für Namen
const DefaultMask = "registry.ollama.ai/library/_:latest"

var defaultMask = func() names.Name {
	n := names.Parse(DefaultMask)
	if !n.IsFullyQualified() {
		panic("default mask is not fully qualified")
	}
	return n
}()

// CompleteName macht einen Namen vollständig qualifiziert
func CompleteName(name string) string {
	return names.Merge(names.Parse(name), defaultMask).String()
}

// Registry ist ein Client für Push/Pull-Operationen gegen eine Ollama Registry
type Registry struct {
	// Cache speichert die Modelle. Wenn nil, wird DefaultCache verwendet.
	Cache *blob.DiskCache

	// UserAgent für Requests an die Registry
	UserAgent string

	// Key für Authentifizierung (nur Ed25519 unterstützt)
	Key crypto.PrivateKey

	// HTTPClient für Registry-Requests. Wenn nil, wird http.DefaultClient verwendet.
	HTTPClient *http.Client

	// MaxStreams ist die maximale Anzahl gleichzeitiger Streams.
	// 0 = runtime.GOMAXPROCS, negativ = unbegrenzt
	MaxStreams int

	// ChunkingThreshold ist die maximale Layer-Größe für Single-Request Downloads
	ChunkingThreshold int64

	// Mask für die Vervollständigung von Namen
	Mask string

	// ReadTimeout für Request-Timeouts
	ReadTimeout time.Duration
}

func (r *Registry) readTimeout() time.Duration {
	if r.ReadTimeout > 0 {
		return r.ReadTimeout
	}
	return 1<<63 - 1 // kein Timeout
}

func (r *Registry) cache() (*blob.DiskCache, error) {
	if r.Cache != nil {
		return r.Cache, nil
	}
	return defaultCache()
}

func (r *Registry) parseName(name string) (names.Name, error) {
	mask := defaultMask
	if r.Mask != "" {
		mask = names.Parse(r.Mask)
	}
	n := names.Merge(names.Parse(name), mask)
	if !n.IsFullyQualified() {
		return names.Name{}, fmt.Errorf("%w: %q", ErrNameInvalid, name)
	}
	return n, nil
}

// DefaultRegistry erstellt eine Registry aus der Umgebung
func DefaultRegistry() (*Registry, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	keyPEM, err := os.ReadFile(filepath.Join(home, ".ollama/id_ed25519"))
	if err != nil && errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}

	var rc Registry
	rc.ReadTimeout = 30 * time.Second
	rc.UserAgent = UserAgent()
	rc.Key, err = ssh.ParseRawPrivateKey(keyPEM)
	if err != nil {
		return nil, err
	}
	maxStreams := os.Getenv("OLLAMA_REGISTRY_MAXSTREAMS")
	if maxStreams != "" {
		var err error
		rc.MaxStreams, err = strconv.Atoi(maxStreams)
		if err != nil {
			return nil, fmt.Errorf("invalid OLLAMA_REGISTRY_MAXSTREAMS: %w", err)
		}
	}
	return &rc, nil
}

// UserAgent generiert den User-Agent String
func UserAgent() string {
	buildinfo, _ := debug.ReadBuildInfo()
	version := buildinfo.Main.Version
	if version == "(devel)" {
		version = "v0.0.0"
	}
	return fmt.Sprintf("ollama/%s (%s %s) Go/%s",
		version,
		runtime.GOARCH,
		runtime.GOOS,
		runtime.Version(),
	)
}

func (r *Registry) maxStreams() int {
	return cmp.Or(r.MaxStreams, runtime.GOMAXPROCS(0))
}

func (r *Registry) maxChunkingThreshold() int64 {
	return cmp.Or(r.ChunkingThreshold, DefaultChunkingThreshold)
}

func (r *Registry) client() *http.Client {
	if r.HTTPClient != nil {
		return r.HTTPClient
	}
	return http.DefaultClient
}
