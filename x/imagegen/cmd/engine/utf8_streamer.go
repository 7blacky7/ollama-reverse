//go:build mlx

// utf8_streamer.go - UTF-8 Streaming-Puffer für Token-Ausgabe
// Enthält: utf8Streamer für korrektes Handling von Multi-Byte-Zeichen
package main

import (
	"unicode/utf8"
)

// utf8Streamer buffers decoded text and emits only complete UTF-8 characters.
// This handles cases where tokenizers output partial multi-byte sequences.
type utf8Streamer struct {
	buffer []byte
}

// Write adds decoded text to the buffer and returns complete UTF-8 characters.
func (s *utf8Streamer) Write(text string) string {
	s.buffer = append(s.buffer, text...)

	// Find the last position that ends with a complete UTF-8 character
	validLen := 0
	for i := 0; i < len(s.buffer); {
		r, size := utf8.DecodeRune(s.buffer[i:])
		if r == utf8.RuneError && size == 1 {
			// Invalid or incomplete UTF-8 sequence at this position
			// Check if it could be a valid start of a multi-byte sequence
			if len(s.buffer)-i < 4 {
				// Might be incomplete, keep it in buffer
				break
			}
			// Definitely invalid, skip this byte
			i++
			validLen = i
		} else {
			i += size
			validLen = i
		}
	}

	if validLen == 0 {
		return ""
	}

	result := string(s.buffer[:validLen])
	s.buffer = s.buffer[validLen:]
	return result
}

// Flush returns any remaining buffered bytes (may be incomplete UTF-8).
func (s *utf8Streamer) Flush() string {
	if len(s.buffer) == 0 {
		return ""
	}
	result := string(s.buffer)
	s.buffer = nil
	return result
}
