// Package config - Integration Registry und Selection
//
// Enthält die Registry aller verfügbaren Integrationen:
// - Runner/Editor Interfaces
// - Integration Registry (claude, codex, droid, etc.)
// - selectIntegration: Auswahl-Dialog für Integrationen
// - runIntegration: Startet eine Integration
package config

import (
	"fmt"
	"maps"
	"os"
	"slices"
)

// Runner kann eine Integration mit einem Model ausführen
type Runner interface {
	Run(model string) error
	// String returns the human-readable name of the integration
	String() string
}

// Editor kann Config-Dateien bearbeiten (unterstützt Multi-Model-Auswahl)
type Editor interface {
	// Paths returns the paths to the config files for the integration
	Paths() []string
	// Edit updates the config files for the integration with the given models
	Edit(models []string) error
	// Models returns the models currently configured for the integration
	// TODO(parthsareen): add error return to Models()
	Models() []string
}

// integrations ist die Registry aller verfügbaren Integrationen
var integrations = map[string]Runner{
	"claude":   &Claude{},
	"clawdbot": &Openclaw{},
	"codex":    &Codex{},
	"moltbot":  &Openclaw{},
	"droid":    &Droid{},
	"opencode": &OpenCode{},
	"openclaw": &Openclaw{},
}

// integrationAliases werden im interaktiven Selector versteckt, funktionieren aber als CLI-Argumente
var integrationAliases = map[string]bool{
	"clawdbot": true,
	"moltbot":  true,
}

func selectIntegration() (string, error) {
	if len(integrations) == 0 {
		return "", fmt.Errorf("no integrations available")
	}

	names := slices.Sorted(maps.Keys(integrations))
	var items []selectItem
	for _, name := range names {
		if integrationAliases[name] {
			continue
		}
		r := integrations[name]
		description := r.String()
		if conn, err := loadIntegration(name); err == nil && len(conn.Models) > 0 {
			description = fmt.Sprintf("%s (%s)", r.String(), conn.Models[0])
		}
		items = append(items, selectItem{Name: name, Description: description})
	}

	return selectPrompt("Select integration:", items)
}

func runIntegration(name, modelName string) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}
	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", r, modelName)
	return r.Run(modelName)
}
