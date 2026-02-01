// completion.go - Completion-Handler fuer Image-Generation
// Dieses Modul verarbeitet Completion-Requests und kommuniziert mit dem Subprocess.
package imagegen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
)

// Completion handles image generation requests by forwarding them to the subprocess.
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	// Extract raw image bytes from llm.ImageData slice
	var images [][]byte
	for _, img := range req.Images {
		images = append(images, img.Data)
	}

	// Build request for subprocess
	creq := struct {
		Prompt string   `json:"prompt"`
		Width  int32    `json:"width,omitempty"`
		Height int32    `json:"height,omitempty"`
		Steps  int32    `json:"steps,omitempty"`
		Seed   int64    `json:"seed,omitempty"`
		Images [][]byte `json:"images,omitempty"`
	}{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  req.Steps,
		Seed:   seed,
		Images: images,
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s", strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024) // 16MB max
	for scanner.Scan() {
		// Parse subprocess response (has singular "image" field)
		var raw struct {
			Image   string `json:"image,omitempty"`
			Content string `json:"content,omitempty"`
			Done    bool   `json:"done"`
			Step    int    `json:"step,omitempty"`
			Total   int    `json:"total,omitempty"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			continue
		}

		// Convert to llm.CompletionResponse
		cresp := llm.CompletionResponse{
			Content:    raw.Content,
			Done:       raw.Done,
			Step:       raw.Step,
			TotalSteps: raw.Total,
			Image:      raw.Image,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	return scanner.Err()
}
