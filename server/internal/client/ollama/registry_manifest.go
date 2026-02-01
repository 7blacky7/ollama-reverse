// Package ollama - Manifest und Layer Typen
//
// Diese Datei enthält:
// - Manifest-Struct mit JSON-Marshaling
// - Layer-Struct
// - ResolveLocal() für lokale Manifest-Auflösung
// - Resolve() für Remote-Manifest-Auflösung
// - Unlink() zum Entfernen von Links
package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"os"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/internal/names"
)

// Manifest repräsentiert ein ollama.com/manifest
type Manifest struct {
	Name   string   `json:"-"` // kanonischer Name des Modells
	Data   []byte   `json:"-"` // Rohdaten des Manifests
	Layers []*Layer `json:"layers"`

	// Config-Layer für Legacy-Kompatibilität
	Config *Layer `json:"config"`
}

// Layer gibt den Layer mit dem angegebenen Digest zurück, oder nil
func (m *Manifest) Layer(d blob.Digest) *Layer {
	for _, l := range m.Layers {
		if l.Digest == d {
			return l
		}
	}
	return nil
}

// All iteriert über alle Layer (Config + Layers)
func (m *Manifest) All() iter.Seq[*Layer] {
	return func(yield func(*Layer) bool) {
		if !yield(m.Config) {
			return
		}
		for _, l := range m.Layers {
			if !yield(l) {
				return
			}
		}
	}
}

// Size berechnet die Gesamtgröße aller Layer
func (m *Manifest) Size() int64 {
	var size int64
	if m.Config != nil {
		size += m.Config.Size
	}
	for _, l := range m.Layers {
		size += l.Size
	}
	return size
}

// MarshalJSON implementiert json.Marshaler
// Fügt ein leeres Config-Objekt hinzu (Registry-Anforderung)
func (m Manifest) MarshalJSON() ([]byte, error) {
	type M Manifest
	v := struct {
		M
		Config Layer `json:"config"`
	}{
		M: M(m),
	}
	return json.Marshal(v)
}

// unmarshalManifest deserialisiert Daten in ein Manifest
// Panics wenn der Name nicht vollständig qualifiziert ist
func unmarshalManifest(n names.Name, data []byte) (*Manifest, error) {
	if !n.IsFullyQualified() {
		panic(fmt.Sprintf("unmarshalManifest: name is not fully qualified: %s", n.String()))
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	m.Name = n.String()
	m.Data = data
	return &m, nil
}

// Layer ist ein Layer in einem Modell
type Layer struct {
	Digest    blob.Digest `json:"digest"`
	MediaType string      `json:"mediaType"`
	Size      int64       `json:"size"`
}

// ResolveLocal löst einen Namen zu einem Manifest im lokalen Cache auf
func (r *Registry) ResolveLocal(name string) (*Manifest, error) {
	_, n, d, err := r.parseNameExtended(name)
	if err != nil {
		return nil, err
	}
	c, err := r.cache()
	if err != nil {
		return nil, err
	}
	if !d.IsValid() {
		// Kein Digest, löse Manifest nach Namen auf
		d, err = c.Resolve(n.String())
		if err != nil {
			return nil, err
		}
	}
	data, err := os.ReadFile(c.GetFile(d))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrModelNotFound, name)
		}
		return nil, err
	}
	m, err := unmarshalManifest(n, data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, errors.Join(ErrManifestInvalid, err))
	}
	return m, nil
}

// Resolve löst einen Namen zu einem Manifest in der Remote-Registry auf
func (r *Registry) Resolve(ctx context.Context, name string) (*Manifest, error) {
	scheme, n, d, err := r.parseNameExtended(name)
	if err != nil {
		return nil, err
	}

	manifestURL := fmt.Sprintf("%s://%s/v2/%s/%s/manifests/%s", scheme, n.Host(), n.Namespace(), n.Model(), n.Tag())
	if d.IsValid() {
		manifestURL = fmt.Sprintf("%s://%s/v2/%s/%s/blobs/%s", scheme, n.Host(), n.Namespace(), n.Model(), d)
	}

	res, err := r.send(ctx, "GET", manifestURL, nil)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	m, err := unmarshalManifest(n, data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, errors.Join(ErrManifestInvalid, err))
	}
	return m, nil
}

// Unlink entfernt einen Link wie blob.DiskCache.Unlink,
// macht aber den Namen vorher vollständig qualifiziert
func (r *Registry) Unlink(name string) (ok bool, _ error) {
	n, err := r.parseName(name)
	if err != nil {
		return false, err
	}
	c, err := r.cache()
	if err != nil {
		return false, err
	}
	return c.Unlink(n.String())
}
