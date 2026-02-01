//go:build windows || darwin

// Modul: inference.go
// Beschreibung: Inference-Compute-Erkennung aus Server-Logs
// Hauptfunktionen:
//   - GetInferenceComputer(): Liest GPU/Compute-Informationen aus dem Server-Log

package server

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"time"
)

// InferenceCompute enthaelt Informationen ueber die verwendete
// Inference-Hardware (GPU, Treiber, VRAM etc.)
type InferenceCompute struct {
	Library string
	Variant string
	Compute string
	Driver  string
	Name    string
	VRAM    string
}

// GetInferenceComputer versucht, Inference-Compute-Informationen aus dem
// Server-Log zu extrahieren. Der ctx-Parameter steuert das Timeout.
func GetInferenceComputer(ctx context.Context) ([]InferenceCompute, error) {
	inference := []InferenceCompute{}
	marker := regexp.MustCompile(`inference compute.*library=`)
	q := `inference compute.*%s=["]([^"]*)["]`
	nq := `inference compute.*%s=(\S+)\s`

	type regex struct {
		q  *regexp.Regexp
		nq *regexp.Regexp
	}

	regexes := map[string]regex{
		"library": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "library")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "library")),
		},
		"variant": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "variant")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "variant")),
		},
		"compute": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "compute")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "compute")),
		},
		"driver": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "driver")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "driver")),
		},
		"name": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "name")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "name")),
		},
		"total": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "total")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "total")),
		},
	}

	get := func(field, line string) string {
		regex, ok := regexes[field]
		if !ok {
			slog.Warn("missing field", "field", field)
			return ""
		}
		match := regex.q.FindStringSubmatch(line)
		if len(match) > 1 {
			return match[1]
		}
		match = regex.nq.FindStringSubmatch(line)
		if len(match) > 1 {
			return match[1]
		}
		return ""
	}

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("timeout scanning server log for inference compute details")
		default:
		}

		file, err := os.Open(serverLogPath)
		if err != nil {
			slog.Debug("failed to open server log", "log", serverLogPath, "error", err)
			time.Sleep(time.Second)
			continue
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			match := marker.FindStringSubmatch(line)
			if len(match) > 0 {
				ic := InferenceCompute{
					Library: get("library", line),
					Variant: get("variant", line),
					Compute: get("compute", line),
					Driver:  get("driver", line),
					Name:    get("name", line),
					VRAM:    get("total", line),
				}

				slog.Info("Matched", "inference compute", ic)
				inference = append(inference, ic)
			} else {
				// Abbruch bei erster nicht-passender Zeile nach Treffern
				if len(inference) > 0 {
					return inference, nil
				}
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
}
