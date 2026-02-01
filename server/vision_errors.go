// MODUL: vision_errors
// ZWECK: Fehler-Definitionen und Error-Handler fuer die Vision API
// INPUT: Fehler, HTTP ResponseWriter, Status-Code
// OUTPUT: JSON-formatierte Fehler-Responses
// NEBENEFFEKTE: HTTP-Responses schreiben
// ABHAENGIGKEITEN: errors, net/http, encoding/json (Standardbibliothek)
// HINWEISE: Fehler-Codes folgen dem API-Schema aus types_siglip.go
package server

import (
	"encoding/json"
	"errors"
	"net/http"
)

// ============================================================================
// Vision API Fehler-Definitionen
// ============================================================================

var (
	// ErrVisionModelNotLoaded wird geworfen wenn das Vision-Modell nicht geladen ist
	ErrVisionModelNotLoaded = errors.New("vision model not loaded")

	// ErrVisionInvalidImage wird geworfen bei ungueltigen Bild-Daten
	ErrVisionInvalidImage = errors.New("invalid image data")

	// ErrVisionEncodingFailed wird geworfen wenn Encoding fehlschlaegt
	ErrVisionEncodingFailed = errors.New("encoding failed")

	// ErrVisionModelNotFound wird geworfen wenn der Modell-Typ nicht gefunden wird
	ErrVisionModelNotFound = errors.New("model type not found")

	// ErrVisionBatchTooLarge wird geworfen wenn die Batch-Groesse das Limit ueberschreitet
	ErrVisionBatchTooLarge = errors.New("batch size exceeds limit")

	// ErrVisionInvalidBase64 wird geworfen bei ungueltiger Base64-Kodierung
	ErrVisionInvalidBase64 = errors.New("invalid base64 encoding")

	// ErrVisionUnsupportedFormat wird geworfen bei nicht unterstuetztem Bildformat
	ErrVisionUnsupportedFormat = errors.New("unsupported image format")
)

// ============================================================================
// Strukturierter API-Fehler
// ============================================================================

// VisionAPIError repraesentiert einen strukturierten API-Fehler.
type VisionAPIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Error implementiert das error Interface.
func (e VisionAPIError) Error() string {
	return e.Message
}

// ============================================================================
// Fehler-Code Mapping
// ============================================================================

// visionErrorCodes mappt Standard-Fehler auf API-Codes
var visionErrorCodes = map[error]string{
	ErrVisionModelNotLoaded:    "MODEL_NOT_LOADED",
	ErrVisionInvalidImage:      "INVALID_IMAGE",
	ErrVisionEncodingFailed:    "ENCODING_ERROR",
	ErrVisionModelNotFound:     "MODEL_NOT_FOUND",
	ErrVisionBatchTooLarge:     "BATCH_TOO_LARGE",
	ErrVisionInvalidBase64:     "INVALID_BASE64",
	ErrVisionUnsupportedFormat: "UNSUPPORTED_FORMAT",
}

// getErrorCode gibt den API-Code fuer einen Fehler zurueck.
func getErrorCode(err error) string {
	// Direkte Suche
	if code, ok := visionErrorCodes[err]; ok {
		return code
	}

	// Wrapped Errors pruefen
	for knownErr, code := range visionErrorCodes {
		if errors.Is(err, knownErr) {
			return code
		}
	}

	return "INTERNAL_ERROR"
}

// ============================================================================
// HTTP Response Helper
// ============================================================================

// writeVisionError schreibt einen Fehler als JSON Response.
func writeVisionError(w http.ResponseWriter, err error, status int) {
	code := getErrorCode(err)
	message := err.Error()

	// VisionAPIError extrahieren falls vorhanden
	var apiErr VisionAPIError
	if errors.As(err, &apiErr) {
		code = apiErr.Code
		message = apiErr.Message
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	response := VisionAPIError{
		Code:    code,
		Message: message,
	}

	json.NewEncoder(w).Encode(response)
}

// NewVisionAPIError erstellt einen neuen strukturierten API-Fehler.
func NewVisionAPIError(code, message string) VisionAPIError {
	return VisionAPIError{
		Code:    code,
		Message: message,
	}
}
