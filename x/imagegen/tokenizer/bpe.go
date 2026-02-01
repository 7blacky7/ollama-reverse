//go:build mlx

// bpe.go - BPE und WordPiece Encoding-Algorithmen
//
// Enthält:
// - encodeBPEMerge: BPE Merge-Algorithmus (GPT-2, SentencePiece)
// - encodeWordPieceInto: WordPiece Algorithmus (BERT)

package tokenizer

import (
	"strings"
)

// encodeChunkInto appends encoded tokens to ids and returns the extended slice
func (t *Tokenizer) encodeChunkInto(s string, ids []int32) []int32 {
	if t.typ == TokenizerWordPiece {
		return t.encodeWordPieceInto(s, ids)
	}

	if s == "" {
		return ids
	}

	// Apply encoding transformation
	// SentencePiece: replace space with ▁
	// BPE: convert bytes using precomputed table (GPT-2 byte-level encoding)
	var encoded string
	if t.typ == TokenizerSentencePiece {
		encoded = strings.ReplaceAll(s, " ", "▁")
	} else {
		var sb strings.Builder
		sb.Grow(len(s) * 2)
		for i := 0; i < len(s); i++ {
			sb.WriteRune(byteToRune[s[i]])
		}
		encoded = sb.String()
	}

	// Fast path: check if entire chunk is a single token
	if id, ok := t.vocab.Reverse[encoded]; ok {
		return append(ids, id)
	}

	return t.encodeBPEMerge(encoded, ids)
}

// encodeBPEMerge encodes using BPE merge algorithm.
// Repeatedly merges the pair with lowest rank until no more merges possible.
func (t *Tokenizer) encodeBPEMerge(encoded string, ids []int32) []int32 {
	// Start with individual runes as parts
	runes := []rune(encoded)
	parts := make([]string, len(runes))
	for i, r := range runes {
		parts[i] = string(r)
	}

	// Repeatedly merge lowest-rank pair
	for len(parts) > 1 {
		minRank := int(0x7FFFFFFF)
		minIdx := -1

		for i := 0; i < len(parts)-1; i++ {
			mergeKey := parts[i] + " " + parts[i+1]
			if rank, ok := t.vocab.Merges[mergeKey]; ok {
				if rank < minRank {
					minRank = rank
					minIdx = i
				}
			}
		}

		if minIdx < 0 {
			break // No more merges possible
		}

		// Merge the pair
		parts[minIdx] = parts[minIdx] + parts[minIdx+1]
		parts = append(parts[:minIdx+1], parts[minIdx+2:]...)
	}

	// Convert parts to token IDs
	for _, part := range parts {
		if id, ok := t.vocab.Reverse[part]; ok {
			ids = append(ids, id)
		} else {
			// Byte fallback for unknown parts
			for _, b := range []byte(part) {
				if id := t.vocab.byteTokens[b]; id >= 0 {
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

// encodeWordPieceInto appends WordPiece tokens to ids and returns extended slice
// Uses greedy longest-match with ## prefix for continuation tokens
func (t *Tokenizer) encodeWordPieceInto(s string, ids []int32) []int32 {
	if s == "" {
		return ids
	}

	// Check if entire string is in vocabulary (common case)
	if id, ok := t.vocab.Reverse[s]; ok {
		return append(ids, id)
	}

	runes := []rune(s)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false

		// Greedy longest-match
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				// Continuation token: prefix with ##
				substr = "##" + substr
			}

			if id, ok := t.vocab.Reverse[substr]; ok {
				ids = append(ids, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			// No match found - use [UNK] token or skip
			if t.unkToken >= 0 {
				ids = append(ids, t.unkToken)
			}
			start++
		}
	}

	return ids
}
