// interactive_utils.go - Hilfsfunktionen fuer den interaktiven Modus
// Enthaelt Datei-Extraktion, Bildverarbeitung und Model-Erstellung
package cmd

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// NewCreateRequest erstellt einen CreateRequest aus runOptions
func NewCreateRequest(name string, opts runOptions) *api.CreateRequest {
	parentModel := opts.ParentModel

	modelName := model.ParseName(parentModel)
	if !modelName.IsValid() {
		parentModel = ""
	}

	req := &api.CreateRequest{
		Model: name,
		From:  cmp.Or(parentModel, opts.Model),
	}

	if opts.System != "" {
		req.System = opts.System
	}

	if len(opts.Options) > 0 {
		req.Parameters = opts.Options
	}

	if len(opts.Messages) > 0 {
		req.Messages = opts.Messages
	}

	return req
}

// normalizeFilePath normalisiert escapte Zeichen in Dateipfaden
func normalizeFilePath(fp string) string {
	return strings.NewReplacer(
		"\\ ", " ", // Escaped space
		"\\(", "(", // Escaped left parenthesis
		"\\)", ")", // Escaped right parenthesis
		"\\[", "[", // Escaped left square bracket
		"\\]", "]", // Escaped right square bracket
		"\\{", "{", // Escaped left curly brace
		"\\}", "}", // Escaped right curly brace
		"\\$", "$", // Escaped dollar sign
		"\\&", "&", // Escaped ampersand
		"\\;", ";", // Escaped semicolon
		"\\'", "'", // Escaped single quote
		"\\\\", "\\", // Escaped backslash
		"\\*", "*", // Escaped asterisk
		"\\?", "?", // Escaped question mark
		"\\~", "~", // Escaped tilde
	).Replace(fp)
}

// extractFileNames extrahiert Bilddatei-Pfade aus einer Eingabe
func extractFileNames(input string) []string {
	// Regex to match file paths starting with optional drive letter, / ./ \ or .\
	// and include escaped or unescaped spaces (\ or %20)
	// and followed by more characters and a file extension
	// This will capture non filename strings, but we'll check for file existence to remove mismatches
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp)\b`
	re := regexp.MustCompile(regexPattern)

	return re.FindAllString(input, -1)
}

// extractFileData extrahiert Bilddaten aus einer Eingabe
func extractFileData(input string) (string, []api.ImageData, error) {
	filePaths := extractFileNames(input)
	var imgs []api.ImageData

	for _, fp := range filePaths {
		nfp := normalizeFilePath(fp)
		data, err := getImageData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "Couldn't process image: %q\n", err)
			return "", imgs, err
		}
		fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, "'"+nfp+"'", "")
		input = strings.ReplaceAll(input, "'"+fp+"'", "")
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return strings.TrimSpace(input), imgs, nil
}

// getImageData liest und validiert Bilddaten aus einer Datei
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

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	// Check if the file size exceeds 100MB
	var maxSize int64 = 100 * 1024 * 1024 // 100MB in bytes
	if info.Size() > maxSize {
		return nil, errors.New("file size exceeds maximum limit (100MB)")
	}

	buf = make([]byte, info.Size())
	_, err = file.Seek(0, 0)
	if err != nil {
		return nil, err
	}

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}
