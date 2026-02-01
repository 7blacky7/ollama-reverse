// run_auth.go - Authentifizierung und Modell-Prüfung
//
// Dieses Modul enthält:
// - waitForOllamaSignin: Zeigt Signin-URL und pollt bis Auth abgeschlossen
// - checkModelCapabilities: Prüft ob Modell Tools unterstützt
// - Token-Limit-Konstanten und Truncation-Logik

package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Token-Limit Konstanten für Tool-Output-Capping
const (
	// localModelTokenLimit ist das Token-Limit für lokale Modelle (kleinerer Kontext).
	localModelTokenLimit = 4000

	// defaultTokenLimit ist das Token-Limit für Cloud/Remote-Modelle.
	defaultTokenLimit = 10000

	// charsPerToken ist eine grobe Schätzung von Zeichen pro Token.
	// TODO: Tokens genauer schätzen mit Tokenizer falls verfügbar
	charsPerToken = 4
)

// isLocalModel prüft ob das Modell lokal läuft (kein Cloud-Modell).
// TODO: Lokale/Cloud-Modell-Identifikation verbessern - könnte Modell-Metadaten prüfen
func isLocalModel(modelName string) bool {
	return !strings.HasSuffix(modelName, "-cloud")
}

// isLocalServer prüft ob Verbindung zu lokalem Ollama-Server besteht.
// TODO: Könnte auch andere Indikatoren für lokal vs. Cloud prüfen
func isLocalServer() bool {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		return true // Default ist localhost:11434
	}

	// URL parsen um Host zu prüfen
	parsed, err := url.Parse(host)
	if err != nil {
		return true // Falls nicht parsebar, lokal annehmen
	}

	hostname := parsed.Hostname()
	return hostname == "localhost" || hostname == "127.0.0.1" || strings.Contains(parsed.Host, ":11434")
}

// truncateToolOutput kürzt Tool-Output um Kontext-Überlauf zu verhindern.
// Verwendet kleineres Limit (4k Tokens) für lokale Modelle, größeres (10k) für Cloud/Remote.
func truncateToolOutput(output, modelName string) string {
	var tokenLimit int
	if isLocalModel(modelName) && isLocalServer() {
		tokenLimit = localModelTokenLimit
	} else {
		tokenLimit = defaultTokenLimit
	}

	maxChars := tokenLimit * charsPerToken
	if len(output) > maxChars {
		return output[:maxChars] + "\n... (output truncated)"
	}
	return output
}

// waitForOllamaSignin zeigt die Signin-URL und pollt bis Authentifizierung abgeschlossen ist.
func waitForOllamaSignin(ctx context.Context) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Signin-URL vom initialen Whoami-Aufruf holen
	_, err = client.Whoami(ctx)
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.SigninURL != "" {
			fmt.Fprintf(os.Stderr, "\n  To sign in, navigate to:\n")
			fmt.Fprintf(os.Stderr, "      %s\n\n", aErr.SigninURL)
			fmt.Fprintf(os.Stderr, "  \033[90mwaiting for sign in to complete...\033[0m")

			// Pollen bis Auth erfolgreich
			ticker := time.NewTicker(2 * time.Second)
			defer ticker.Stop()

			for {
				select {
				case <-ctx.Done():
					fmt.Fprintf(os.Stderr, "\n")
					return ctx.Err()
				case <-ticker.C:
					user, whoamiErr := client.Whoami(ctx)
					if whoamiErr == nil && user != nil && user.Name != "" {
						fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K  \033[1msigned in:\033[0m %s\n", user.Name)
						return nil
					}
					// Noch warten, Punkt anzeigen
					fmt.Fprintf(os.Stderr, ".")
				}
			}
		}
		return err
	}
	return nil
}

// checkModelCapabilities prüft ob das Modell Tools unterstützt.
func checkModelCapabilities(ctx context.Context, modelName string) (supportsTools bool, err error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return false, err
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return false, err
	}

	for _, cap := range resp.Capabilities {
		if cap == model.CapabilityTools {
			return true, nil
		}
	}

	return false, nil
}
