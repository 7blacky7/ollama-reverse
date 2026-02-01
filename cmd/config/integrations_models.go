// Package config - Model Selection für Integrationen
//
// Funktionen zur Model-Auswahl:
// - selectModels: Interaktive Model-Auswahl
// - Cloud Model Authentifizierung
// - Sign-in Flow mit Browser-Redirect
package config

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// selectModels lässt den Benutzer Models für eine Integration auswählen
func selectModels(ctx context.Context, name, current string) ([]string, error) {
	r, ok := integrations[name]
	if !ok {
		return nil, fmt.Errorf("unknown integration: %s", name)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	if len(models.Models) == 0 {
		return nil, fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	items, cloudModels := buildModelItems(models)
	if len(items) == 0 {
		return nil, fmt.Errorf("no local models available, run 'ollama pull <model>' first")
	}

	preChecked := getPreCheckedModels(name, r)
	checked := make(map[string]bool, len(preChecked))
	for _, n := range preChecked {
		checked[n] = true
	}

	current = resolveCurrentModel(current, items)
	if checked[current] {
		preChecked = append([]string{current}, slices.DeleteFunc(preChecked, func(m string) bool { return m == current })...)
	}

	sortModelItems(items, checked)

	selected, err := promptForModels(r, items, preChecked)
	if err != nil {
		return nil, err
	}

	return handleCloudModels(ctx, client, selected, cloudModels)
}

func buildModelItems(models *api.ListResponse) ([]selectItem, map[string]bool) {
	var items []selectItem
	cloudModels := make(map[string]bool)
	for _, m := range models.Models {
		if m.RemoteModel != "" {
			cloudModels[m.Name] = true
		}
		items = append(items, selectItem{Name: m.Name})
	}
	return items, cloudModels
}

func getPreCheckedModels(name string, r Runner) []string {
	if saved, err := loadIntegration(name); err == nil {
		return saved.Models
	} else if editor, ok := r.(Editor); ok {
		return editor.Models()
	}
	return nil
}

func resolveCurrentModel(current string, items []selectItem) string {
	for _, item := range items {
		if item.Name == current || strings.HasPrefix(item.Name, current+":") {
			return item.Name
		}
	}
	return current
}

func sortModelItems(items []selectItem, checked map[string]bool) {
	slices.SortFunc(items, func(a, b selectItem) int {
		ac, bc := checked[a.Name], checked[b.Name]
		if ac != bc {
			if ac {
				return -1
			}
			return 1
		}
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})
}

func promptForModels(r Runner, items []selectItem, preChecked []string) ([]string, error) {
	if _, ok := r.(Editor); ok {
		return multiSelectPrompt(fmt.Sprintf("Select models for %s:", r), items, preChecked)
	}
	model, err := selectPrompt(fmt.Sprintf("Select model for %s:", r), items)
	if err != nil {
		return nil, err
	}
	return []string{model}, nil
}

func handleCloudModels(ctx context.Context, client *api.Client, selected []string, cloudModels map[string]bool) ([]string, error) {
	var selectedCloudModels []string
	for _, m := range selected {
		if cloudModels[m] {
			selectedCloudModels = append(selectedCloudModels, m)
		}
	}

	if len(selectedCloudModels) == 0 {
		return selected, nil
	}

	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return selected, nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		return nil, err
	}

	return handleSignIn(ctx, client, selected, selectedCloudModels, aErr.SigninURL)
}

func handleSignIn(ctx context.Context, client *api.Client, selected, cloudModels []string, signinURL string) ([]string, error) {
	modelList := strings.Join(cloudModels, ", ")
	yes, err := confirmPrompt(fmt.Sprintf("sign in to use %s?", modelList))
	if err != nil || !yes {
		return nil, fmt.Errorf("%s requires sign in", modelList)
	}

	fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", signinURL)
	openBrowser(signinURL)

	return waitForSignIn(ctx, client, selected)
}

func openBrowser(url string) {
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("open", url).Start()
	case "linux":
		_ = exec.Command("xdg-open", url).Start()
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	}
}

func waitForSignIn(ctx context.Context, client *api.Client, selected []string) ([]string, error) {
	spinnerFrames := []string{"|", "/", "-", "\\"}
	frame := 0

	fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[0])

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\r\033[K")
			return nil, ctx.Err()
		case <-ticker.C:
			frame++
			fmt.Fprintf(os.Stderr, "\r\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])

			if frame%10 == 0 {
				u, err := client.Whoami(ctx)
				if err == nil && u != nil && u.Name != "" {
					fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", u.Name)
					return selected, nil
				}
			}
		}
	}
}
