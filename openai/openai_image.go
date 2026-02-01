// openai_image.go - Bild-bezogene Konvertierungsfunktionen
//
// Enthaelt:
// - decodeImageURL: Base64 Data-URI in Bilddaten dekodieren
// - FromImageGenerationRequest: Bild-Generation Request konvertieren
// - FromImageEditRequest: Bild-Edit Request konvertieren
//
// Verwandte Dateien:
// - openai_types.go: Typdefinitionen (ImageGenerationRequest, etc.)
// - openai_to.go: ToImageGenerationResponse
// - openai_from.go: Andere From*-Konvertierungen
package openai

import (
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// decodeImageURL dekodiert eine base64 Data-URI in rohe Bilddaten
func decodeImageURL(url string) (api.ImageData, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return nil, errors.New("image URLs are not currently supported, please use base64 encoded data instead")
	}

	types := []string{"jpeg", "jpg", "png", "webp"}

	// Unterstuetzung fuer leeren MIME-Typ (wie bei /api/chat mit reinem base64)
	if strings.HasPrefix(url, "data:;base64,") {
		url = strings.TrimPrefix(url, "data:;base64,")
	} else {
		valid := false
		for _, t := range types {
			prefix := "data:image/" + t + ";base64,"
			if strings.HasPrefix(url, prefix) {
				url = strings.TrimPrefix(url, prefix)
				valid = true
				break
			}
		}
		if !valid {
			return nil, errors.New("invalid image input")
		}
	}

	img, err := base64.StdEncoding.DecodeString(url)
	if err != nil {
		return nil, errors.New("invalid image input")
	}
	return img, nil
}

// FromImageGenerationRequest konvertiert einen OpenAI Image-Generation Request zu Ollama GenerateRequest
func FromImageGenerationRequest(r ImageGenerationRequest) api.GenerateRequest {
	req := api.GenerateRequest{
		Model:  r.Model,
		Prompt: r.Prompt,
	}
	// Groesse parsen falls angegeben (z.B. "1024x768")
	if r.Size != "" {
		var w, h int32
		if _, err := fmt.Sscanf(r.Size, "%dx%d", &w, &h); err == nil {
			req.Width = w
			req.Height = h
		}
	}
	if r.Seed != nil {
		if req.Options == nil {
			req.Options = map[string]any{}
		}
		req.Options["seed"] = *r.Seed
	}
	return req
}

// FromImageEditRequest konvertiert einen OpenAI Image-Edit Request zu Ollama GenerateRequest
func FromImageEditRequest(r ImageEditRequest) (api.GenerateRequest, error) {
	req := api.GenerateRequest{
		Model:  r.Model,
		Prompt: r.Prompt,
	}

	// Eingabebild dekodieren
	if r.Image != "" {
		imgData, err := decodeImageURL(r.Image)
		if err != nil {
			return api.GenerateRequest{}, fmt.Errorf("invalid image: %w", err)
		}
		req.Images = append(req.Images, imgData)
	}

	// Groesse parsen falls angegeben (z.B. "1024x768")
	if r.Size != "" {
		var w, h int32
		if _, err := fmt.Sscanf(r.Size, "%dx%d", &w, &h); err == nil {
			req.Width = w
			req.Height = h
		}
	}

	if r.Seed != nil {
		if req.Options == nil {
			req.Options = map[string]any{}
		}
		req.Options["seed"] = *r.Seed
	}

	return req, nil
}
