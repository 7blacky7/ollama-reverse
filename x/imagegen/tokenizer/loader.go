//go:build mlx

// loader.go - Tokenizer Laden und Parsen (tokenizer.json Format)
//
// Enthält:
// - Load: Lädt aus Datei oder Verzeichnis
// - LoadFromBytes, LoadFromBytesWithConfig: Laden aus Byte-Slices
// - loadFromTokenizerJSON: Parst tokenizer.json Format
// - detectSentencePiece, extractPretokenizer, rewritePatternForRE2
//
// Siehe auch: loader_vocab.go für GPT-style vocab.json + merges.txt

package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// LoadFromBytes loads a tokenizer from tokenizer.json bytes.
// This is useful when loading from blob storage where the file content is already in memory.
// Note: This won't load special token config from companion files. Use LoadFromBytesWithConfig
// to provide tokenizer_config.json data for proper PAD/EOS token loading.
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	return loadFromTokenizerJSON(data, "")
}

// LoadFromBytesWithConfig loads a tokenizer from tokenizer.json bytes with additional config files.
// This is useful when loading from blob storage where companion config files are also blobs.
func LoadFromBytesWithConfig(data []byte, config *TokenizerConfig) (*Tokenizer, error) {
	t, err := loadFromTokenizerJSON(data, "")
	if err != nil {
		return nil, err
	}

	if config == nil {
		return t, nil
	}

	// Apply special token configs from provided data
	loadSpecialTokenConfigFromBytes(t, config)

	return t, nil
}

// Load loads a tokenizer from a path which can be:
// - A tokenizer.json file
// - A directory containing tokenizer.json or vocab.json + merges.txt
func Load(path string) (*Tokenizer, error) {
	// Check if path is a directory
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		dir := strings.TrimSuffix(path, "/") + "/"
		// Try tokenizer.json first
		if data, err := os.ReadFile(dir + "tokenizer.json"); err == nil {
			return loadFromTokenizerJSON(data, dir)
		}
		// Fall back to vocab.json + merges.txt
		return LoadVocabMerges(path)
	}

	// It's a file - read it directly
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer: %w", err)
	}

	// Get directory for loading companion files
	dir := ""
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		dir = path[:idx+1]
	}
	return loadFromTokenizerJSON(data, dir)
}

// loadFromTokenizerJSON parses a tokenizer.json file
func loadFromTokenizerJSON(data []byte, dir string) (*Tokenizer, error) {

	var raw struct {
		Model struct {
			Type   string           `json:"type"` // "BPE" or "WordPiece"
			Vocab  map[string]int32 `json:"vocab"`
			Merges json.RawMessage  `json:"merges"` // Can be []string or [][]string (BPE only)
		} `json:"model"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		Decoder      json.RawMessage `json:"decoder"`
		AddedTokens  []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer: %w", err)
	}

	// Parse merges - can be []string (Llama) or [][]string (GPT-OSS)
	// WordPiece models don't have merges
	var mergesStrings []string
	if raw.Model.Type != "WordPiece" && raw.Model.Merges != nil {
		var mergesArrays [][]string
		if err := json.Unmarshal(raw.Model.Merges, &mergesStrings); err != nil {
			// Try array of arrays format
			if err := json.Unmarshal(raw.Model.Merges, &mergesArrays); err != nil {
				return nil, fmt.Errorf("failed to parse merges: %w", err)
			}
			// Convert [][]string to []string
			mergesStrings = make([]string, len(mergesArrays))
			for i, pair := range mergesArrays {
				mergesStrings[i] = pair[0] + " " + pair[1]
			}
		}
	}

	// Build tokenizer
	t := &Tokenizer{
		vocab: &Vocabulary{
			Values:  make([]string, len(raw.Model.Vocab)),
			Reverse: raw.Model.Vocab,
			Merges:  make(map[string]int, len(mergesStrings)),
			BOS:     -1,
			PAD:     -1,
		},
		specialTokens: make(map[string]int32),
	}

	// Build values array
	for token, id := range raw.Model.Vocab {
		if int(id) >= len(t.vocab.Values) {
			newValues := make([]string, id+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[id] = token
	}

	// Build merges map
	for i, merge := range mergesStrings {
		t.vocab.Merges[merge] = i
	}

	// Add all added_tokens to vocabulary and special tokens map.
	for _, tok := range raw.AddedTokens {
		if int(tok.ID) >= len(t.vocab.Values) {
			newValues := make([]string, tok.ID+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[tok.ID] = tok.Content
		t.specialTokens[tok.Content] = tok.ID
	}

	// Load special token configuration from companion files
	loadSpecialTokenConfig(dir, t)

	// Precompute byte token IDs for <0xNN> fallback
	initByteTokens(t)

	// Determine tokenizer type
	switch {
	case raw.Model.Type == "WordPiece":
		t.typ = TokenizerWordPiece
	case detectSentencePiece(raw.Decoder):
		t.typ = TokenizerSentencePiece
	default:
		t.typ = TokenizerBPE
	}

	// Parse and compile pretokenizer pattern (BPE only - SentencePiece doesn't use pretokenizer)
	if t.typ == TokenizerBPE {
		pattern := extractPretokenizer(raw.PreTokenizer)
		if pattern == "" {
			pattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
		}
		re, err := regexp.Compile(rewritePatternForRE2(pattern))
		if err != nil {
			return nil, fmt.Errorf("failed to compile pretokenizer regex %q: %w", pattern, err)
		}
		t.pretokenizer = re
	}

	return t, nil
}

// detectSentencePiece checks if the decoder uses SentencePiece-style (▁ for spaces)
func detectSentencePiece(data json.RawMessage) bool {
	if data == nil {
		return false
	}

	// Check for Sequence decoder with Replace step (SentencePiece style)
	var seq struct {
		Type     string `json:"type"`
		Decoders []struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
		} `json:"decoders"`
	}
	if err := json.Unmarshal(data, &seq); err == nil {
		if seq.Type == "Sequence" {
			for _, dec := range seq.Decoders {
				if dec.Type == "Replace" && dec.Pattern.String == "▁" {
					return true
				}
			}
		}
	}

	// Check for direct ByteLevel decoder (GPT-2 style)
	var simple struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &simple); err == nil {
		if simple.Type == "ByteLevel" {
			return false
		}
	}

	return false
}

// initByteTokens precomputes byte token IDs for <0xNN> fallback encoding
func initByteTokens(t *Tokenizer) {
	for i := range t.vocab.byteTokens {
		t.vocab.byteTokens[i] = -1
	}
	for b := 0; b < 256; b++ {
		token := fmt.Sprintf("<0x%02X>", b)
		if id, ok := t.vocab.Reverse[token]; ok {
			t.vocab.byteTokens[b] = id
		}
	}
}

// extractPretokenizer extracts the regex pattern from the pre_tokenizer config
func extractPretokenizer(data json.RawMessage) string {
	if data == nil {
		return ""
	}

	// Try to parse as a single Split pretokenizer
	var single struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
	}
	if err := json.Unmarshal(data, &single); err == nil && single.Pattern.Regex != "" {
		return single.Pattern.Regex
	}

	// Try to parse as Sequence of pretokenizers - use first Split pattern
	var seq struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(data, &seq); err == nil && seq.Type == "Sequence" {
		for _, pt := range seq.Pretokenizers {
			if pt.Type == "Split" && pt.Pattern.Regex != "" {
				return pt.Pattern.Regex
			}
		}
	}

	return ""
}

// rewritePatternForRE2 rewrites HuggingFace pretokenizer regex patterns to be
// compatible with Go's regexp package (RE2).
func rewritePatternForRE2(pattern string) string {
	// Replace lookahead pattern with simple \s+ - we fix boundaries in encodeWithRegex()
	pattern = strings.ReplaceAll(pattern, `\s+(?!\S)|\s+`, `\s+`)

	// Handle the pattern when it appears with a ? suffix (optional contractions in GPT-4o style)
	pattern = strings.ReplaceAll(pattern,
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?`)

	// Expand case-insensitive contraction pattern to explicit alternations
	pattern = strings.ReplaceAll(pattern,
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)`,
		`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`)

	return pattern
}

