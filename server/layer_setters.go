// layer_setters.go - Layer-Setter-Funktionen fuer Model-Erstellung
//
// Dieses Modul enthaelt:
// - setTemplate: Setzt den Template-Layer
// - setSystem: Setzt den System-Layer
// - setLicense: Fuegt Lizenz-Layer hinzu
// - setParameters: Setzt den Parameters-Layer (mit Merge)
// - setMessages: Setzt den Messages-Layer
package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
)

// setTemplate setzt den Template-Layer
// Validiert das Template vor dem Setzen
func setTemplate(layers []manifest.Layer, t string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.template")

	// Template validieren
	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}

	// Neuen Layer erstellen
	blob := strings.NewReader(t)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.template")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

// setSystem setzt den System-Layer
// Leerer String entfernt den Layer
func setSystem(layers []manifest.Layer, s string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.system")

	if s != "" {
		blob := strings.NewReader(s)
		layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.system")
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// setLicense fuegt einen Lizenz-Layer hinzu
// Mehrfache Aufrufe fuegen mehrere Lizenzen hinzu
func setLicense(layers []manifest.Layer, l string) ([]manifest.Layer, error) {
	blob := strings.NewReader(l)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.license")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

// setParameters setzt den Parameters-Layer
// Merged neue Parameter mit existierenden (neue haben Vorrang)
func setParameters(layers []manifest.Layer, p map[string]any) ([]manifest.Layer, error) {
	if p == nil {
		p = make(map[string]any)
	}

	// Existierende Parameter laden und mergen
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}

		digestPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		fn, err := os.Open(digestPath)
		if err != nil {
			return nil, err
		}
		defer fn.Close()

		var existing map[string]any
		if err := json.NewDecoder(fn).Decode(&existing); err != nil {
			return nil, err
		}

		// Existierende Parameter uebernehmen, wenn nicht ueberschrieben
		for k, v := range existing {
			if _, exists := p[k]; exists {
				continue
			}
			p[k] = v
		}
	}

	// Keine Parameter? Nichts zu tun
	if len(p) == 0 {
		return layers, nil
	}

	// Alten Layer entfernen und neuen erstellen
	layers = removeLayer(layers, "application/vnd.ollama.image.params")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(p); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

// setMessages setzt den Messages-Layer
// Leere Messages behalten alte Messages bei
func setMessages(layers []manifest.Layer, m []api.Message) ([]manifest.Layer, error) {
	// Alte Messages beibehalten wenn keine neuen angegeben
	if len(m) == 0 {
		return layers, nil
	}

	fmt.Printf("removing old messages\n")
	layers = removeLayer(layers, "application/vnd.ollama.image.messages")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(m); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.messages")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}
