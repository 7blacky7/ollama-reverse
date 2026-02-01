//go:build mlx

// loader_vocab.go - GPT-Style Tokenizer Laden (vocab.json + merges.txt)
//
// Enthält:
// - LoadVocabMerges: Lädt GPT-2/tiktoken Format aus separaten Dateien

package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// LoadVocabMerges loads a tokenizer from vocab.json + merges.txt format (GPT-style)
func LoadVocabMerges(dir string) (*Tokenizer, error) {
	vocabPath := dir + "/vocab.json"
	mergesPath := dir + "/merges.txt"
	addedTokensPath := dir + "/added_tokens.json"

	// Load vocab
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vocab.json: %w", err)
	}

	vocabMap := make(map[string]int32)
	if err := json.Unmarshal(vocabData, &vocabMap); err != nil {
		return nil, fmt.Errorf("failed to parse vocab.json: %w", err)
	}

	// Load merges
	mergesData, err := os.ReadFile(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read merges.txt: %w", err)
	}

	mergesLines := strings.Split(string(mergesData), "\n")
	var mergesStrings []string
	for _, line := range mergesLines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		mergesStrings = append(mergesStrings, line)
	}

	// Build tokenizer
	t := &Tokenizer{
		vocab: &Vocabulary{
			Values:  make([]string, len(vocabMap)),
			Reverse: vocabMap,
			Merges:  make(map[string]int, len(mergesStrings)),
			BOS:     -1,
			PAD:     -1,
		},
		specialTokens: make(map[string]int32),
	}

	// Load added tokens if exists
	if addedData, err := os.ReadFile(addedTokensPath); err == nil {
		addedMap := make(map[string]int32)
		if err := json.Unmarshal(addedData, &addedMap); err == nil {
			for token, id := range addedMap {
				vocabMap[token] = id
				t.specialTokens[token] = id
			}
		}
	}

	// Build values array
	for token, id := range vocabMap {
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

	// Load special token configuration from companion files
	loadSpecialTokenConfig(dir+"/", t)

	// Precompute byte token IDs for <0xNN> fallback
	initByteTokens(t)

	// GPT-2/tiktoken pretokenizer pattern
	pattern := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	re, err := regexp.Compile(rewritePatternForRE2(pattern))
	if err != nil {
		return nil, fmt.Errorf("failed to compile pretokenizer regex: %w", err)
	}
	t.pretokenizer = re

	return t, nil
}
