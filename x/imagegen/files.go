// files.go - Datei-Extraktion und Terminal-Display-Funktionen.
//
// Dieses Modul enthaelt:
// - Bildpfad-Extraktion aus Prompts
// - Bilddaten-Validierung und -Laden
// - Terminal-Image-Display (iTerm2, Kitty, WezTerm, Ghostty)
// - Dateiname-Sanitization

package imagegen

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

// extractFileNames finds image file paths in the input string.
func extractFileNames(input string) []string {
	// Regex to match file paths with image extensions
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp)\b`
	re := regexp.MustCompile(regexPattern)
	return re.FindAllString(input, -1)
}

// extractFileData extracts image data from file paths found in the input.
// Returns the cleaned prompt (with file paths removed) and the image data.
func extractFileData(input string) (string, []api.ImageData, error) {
	filePaths := extractFileNames(input)
	var imgs []api.ImageData

	for _, fp := range filePaths {
		// Normalize shell escapes
		nfp := strings.ReplaceAll(fp, "\\ ", " ")
		nfp = strings.ReplaceAll(nfp, "\\(", "(")
		nfp = strings.ReplaceAll(nfp, "\\)", ")")
		nfp = strings.ReplaceAll(nfp, "%20", " ")

		data, err := getImageData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return "", nil, err
		}
		fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return strings.TrimSpace(input), imgs, nil
}

// getImageData reads and validates image data from a file.
func getImageData(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	buf := make([]byte, 512)
	_, err = file.Read(buf)
	if err != nil {
		return nil, err
	}

	contentType := http.DetectContentType(buf)
	allowedTypes := []string{"image/jpeg", "image/jpg", "image/png", "image/webp"}
	if !slices.Contains(allowedTypes, contentType) {
		return nil, fmt.Errorf("invalid image type: %s", contentType)
	}

	// Re-read the full file
	return os.ReadFile(filePath)
}

// sanitizeFilename removes characters that aren't safe for filenames.
func sanitizeFilename(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, " ", "-")
	// Remove any character that's not alphanumeric or hyphen
	var result strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			result.WriteRune(r)
		}
	}
	return result.String()
}

// displayImageInTerminal attempts to render an image inline in the terminal.
// Supports iTerm2, Kitty, WezTerm, Ghostty, and other terminals with inline image support.
// Returns true if the image was displayed, false otherwise.
func displayImageInTerminal(imagePath string) bool {
	// Check if terminal supports inline images
	termProgram := os.Getenv("TERM_PROGRAM")
	kittyWindowID := os.Getenv("KITTY_WINDOW_ID")
	weztermPane := os.Getenv("WEZTERM_PANE")
	ghostty := os.Getenv("GHOSTTY_RESOURCES_DIR")

	// Read the image file
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return false
	}

	encoded := base64.StdEncoding.EncodeToString(data)

	switch {
	case termProgram == "iTerm.app" || termProgram == "WezTerm" || weztermPane != "":
		// iTerm2/WezTerm inline image protocol
		// ESC ] 1337 ; File = [arguments] : base64 BEL
		fmt.Printf("\033]1337;File=inline=1;preserveAspectRatio=1:%s\a\n", encoded)
		return true

	case kittyWindowID != "" || ghostty != "" || termProgram == "ghostty":
		// Kitty graphics protocol (also used by Ghostty)
		// Send in chunks for large images
		const chunkSize = 4096
		for i := 0; i < len(encoded); i += chunkSize {
			end := min(i+chunkSize, len(encoded))
			chunk := encoded[i:end]

			if i == 0 {
				// First chunk: a=T (transmit), f=100 (PNG), m=1 (more chunks follow) or m=0 (last chunk)
				more := 1
				if end >= len(encoded) {
					more = 0
				}
				fmt.Printf("\033_Ga=T,f=100,m=%d;%s\033\\", more, chunk)
			} else if end >= len(encoded) {
				// Last chunk
				fmt.Printf("\033_Gm=0;%s\033\\", chunk)
			} else {
				// Middle chunk
				fmt.Printf("\033_Gm=1;%s\033\\", chunk)
			}
		}
		fmt.Println()
		return true

	default:
		return false
	}
}
