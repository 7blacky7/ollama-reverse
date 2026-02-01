//go:build mlx

// encode.go - Text zu Token-IDs encodieren
//
// Enthält:
// - Encode: Text zu Token-IDs (parallel für große Inputs)
// - splitBySpecialTokens: Trennt Special Tokens
//
// Siehe auch: bpe.go für Encoding-Algorithmen, decode.go für Decoding

package tokenizer

import (
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// Konstante für parallele Verarbeitung (4KB Schwellwert)
const parallelThreshold = 4096

// isNonNewlineWhitespace returns true if s contains only whitespace characters (no newlines)
func isNonNewlineWhitespace(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r == '\n' || r == '\r' {
			return false
		}
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// splitBySpecialTokens splits text into parts, keeping special tokens as separate elements
func (t *Tokenizer) splitBySpecialTokens(s string) []string {
	if len(t.specialTokens) == 0 {
		return []string{s}
	}

	// Sort special tokens by length (longest first) to match greedily
	tokens := make([]string, 0, len(t.specialTokens))
	for tok := range t.specialTokens {
		tokens = append(tokens, tok)
	}
	sort.Slice(tokens, func(i, j int) bool {
		return len(tokens[i]) > len(tokens[j])
	})

	var result []string
	remaining := s

	for len(remaining) > 0 {
		found := false
		for _, tok := range tokens {
			if strings.HasPrefix(remaining, tok) {
				result = append(result, tok)
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if !found {
			// Find next special token position
			nextPos := len(remaining)
			for _, tok := range tokens {
				if idx := strings.Index(remaining, tok); idx != -1 && idx < nextPos {
					nextPos = idx
				}
			}
			if nextPos > 0 {
				result = append(result, remaining[:nextPos])
			}
			remaining = remaining[nextPos:]
		}
	}

	return result
}

// Encode tokenizes text to token IDs. Parallelizes for large inputs (>4KB).
func (t *Tokenizer) Encode(s string, addBOS bool) []int32 {
	// First: split by special tokens
	parts := t.splitBySpecialTokens(s)

	// Second: collect all pretokenizer chunks
	type chunk struct {
		text      string
		isSpecial bool
	}
	var allChunks []chunk

	if t.pretokenizer != nil {
		re := t.pretokenizer
		for _, part := range parts {
			if _, ok := t.specialTokens[part]; ok {
				allChunks = append(allChunks, chunk{part, true})
				continue
			}

			// Split by pretokenizer regex
			type match struct{ start, end int }
			var matches []match
			offset := 0
			for offset < len(part) {
				loc := re.FindStringIndex(part[offset:])
				if loc == nil {
					break
				}
				matches = append(matches, match{offset + loc[0], offset + loc[1]})
				offset += loc[1]
			}

			// Apply whitespace boundary fix for Python regex compatibility
			for i := 0; i < len(matches)-1; i++ {
				m := part[matches[i].start:matches[i].end]
				next := part[matches[i+1].start:matches[i+1].end]

				if isNonNewlineWhitespace(m) && len(next) > 0 {
					firstRune, _ := utf8.DecodeRuneInString(next)
					if unicode.IsLetter(firstRune) {
						lastSpaceStart := matches[i].end
						for j := matches[i].end; j > matches[i].start; {
							r, size := utf8.DecodeLastRuneInString(part[matches[i].start:j])
							if unicode.IsSpace(r) {
								lastSpaceStart = j - size
								break
							}
							j -= size
						}
						if lastSpaceStart > matches[i].start {
							matches[i].end = lastSpaceStart
							matches[i+1].start = lastSpaceStart
						} else {
							matches[i+1].start = matches[i].start
							matches[i].end = matches[i].start
						}
					}
				}
			}

			for _, m := range matches {
				if m.end > m.start {
					allChunks = append(allChunks, chunk{part[m.start:m.end], false})
				}
			}
		}
	} else {
		// No pretokenizer - treat each part as a single chunk
		for _, part := range parts {
			if _, ok := t.specialTokens[part]; ok {
				allChunks = append(allChunks, chunk{part, true})
			} else {
				allChunks = append(allChunks, chunk{part, false})
			}
		}
	}

	// Encode chunks - parallel for large inputs, sequential otherwise
	var ids []int32
	if len(s) < parallelThreshold {
		for _, c := range allChunks {
			if c.isSpecial {
				if id, ok := t.specialTokens[c.text]; ok {
					ids = append(ids, id)
				}
			} else {
				ids = t.encodeChunkInto(c.text, ids)
			}
		}
	} else {
		numWorkers := runtime.GOMAXPROCS(0)
		if numWorkers > len(allChunks) {
			numWorkers = len(allChunks)
		}

		chunksPer := (len(allChunks) + numWorkers - 1) / numWorkers
		results := make([][]int32, numWorkers)
		var wg sync.WaitGroup

		for i := 0; i < numWorkers; i++ {
			start := i * chunksPer
			end := start + chunksPer
			if end > len(allChunks) {
				end = len(allChunks)
			}
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(i int, chunks []chunk) {
				defer wg.Done()
				var r []int32
				for _, c := range chunks {
					if c.isSpecial {
						if id, ok := t.specialTokens[c.text]; ok {
							r = append(r, id)
						}
					} else {
						r = t.encodeChunkInto(c.text, r)
					}
				}
				results[i] = r
			}(i, allChunks[start:end])
		}
		wg.Wait()

		for _, r := range results {
			ids = append(ids, r...)
		}
	}

	if addBOS && t.vocab.BOS >= 0 {
		ids = append([]int32{t.vocab.BOS}, ids...)
	}
	return ids
}
